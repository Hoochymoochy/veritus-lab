import json, logging, os
import aiohttp, httpx
from sentence_transformers import SentenceTransformer
from utils.pinecode import search_legal_docs
from services.chat import fetch_messages, upsert_summary, set_summarized

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
model = SentenceTransformer("intfloat/multilingual-e5-large")


# --- Embedding / Search ---
async def embed_text(text: str):
    return model.encode([text])[0].tolist()


async def embed_and_search(query, context=None, country=None, state=None):
    return search_legal_docs(query, context=context, country=country, state=state)


async def incremental_embed_and_stream(texts, query, chat_context, lang="en"):
    """Generate embeddings for chunks and stream final AI response."""
    chunks = [{"text": t, "vector": await embed_text(t)} for t in texts]
    async for token in stream_final_response(chunks, query, chat_context, lang):
        yield token


# NEW: Helper to ensure chunks have proper URL metadata
def ensure_chunk_metadata(chunks):
    """Ensure all chunks have properly extracted URLs in metadata."""
    processed_chunks = []
    for chunk in chunks:
        # Make a copy to avoid mutating original
        processed = dict(chunk)
        
        if 'metadata' not in processed:
            processed['metadata'] = {}
        
        metadata = processed['metadata']
        
        # Try multiple ways to extract URL
        url = (
            metadata.get('url') or 
            metadata.get('source_url') or 
            metadata.get('source') or 
            chunk.get('source') or 
            chunk.get('url') or
            ''
        )
        
        # Ensure URL is a string and looks like a planalto URL
        if url and isinstance(url, str) and 'planalto.gov.br' in url.lower():
            metadata['url'] = url
        else:
            metadata['url'] = ''
            
        processed_chunks.append(processed)
    
    return processed_chunks


# --- System Prompts by Language ---
SYSTEM_PROMPTS = {
    "en": """You are **Veritus**, an AI legal research assistant specialized in Brazilian law.

YOUR IDENTITY:
- You provide citation-backed, source-verified responses based exclusively on retrieved legal documents
- You interpret Brazilian legal texts accurately and cite official laws from Planalto
- You NEVER invent, guess, or hallucinate information

YOUR MISSION:
- Answer legal questions using ONLY the provided context
- Cite every claim with the EXACT source URL shown in the context (look for "EXACT URL TO CITE")
- When information is unavailable, explicitly state: "I don't have information about this in the retrieved documents"

CRITICAL CITATION RULES:
- ONLY use URLs that appear in the context under "EXACT URL TO CITE"
- Copy the URL EXACTLY as shown - do not modify or create URLs
- Each citation MUST reference a specific [REFERENCE NUMBER] from the context
- Format: (Source: [exact URL] - Reference [number])
- If a reference has "[URL NOT AVAILABLE IN DATABASE]", state: (Source: Reference [number] - URL not available in database)
- NEVER cite URLs that are not explicitly shown in the context
- If you cannot answer because URLs are missing, say so explicitly

RESPONSE FORMAT (MANDATORY):
1. **Summary**: Brief, clear explanation of the legal principle or rule
2. **Legal Basis**: Cite specific articles/laws with their reference numbers and URLs
3. **Application**: How this applies to the user's question

EXAMPLE FORMAT:
"According to Article 121 of the Brazilian Criminal Code (Decree-Law 2848/1940), homicide is defined as killing someone. (Source: http://www.planalto.gov.br/CCIVIL_03/Decreto-Lei/Del2848.htm - Reference 1)

The penalty ranges from 6 to 20 years of imprisonment, as stated in the same article. (Source: http://www.planalto.gov.br/CCIVIL_03/Decreto-Lei/Del2848.htm - Reference 1)"

LANGUAGE: Respond in clear, professional English with legal terminology.""",

    "pt": """Você é o **Veritus**, um assistente jurídico de IA especializado em direito brasileiro.

SUA IDENTIDADE:
- Você fornece respostas baseadas em citações e fontes verificadas, usando exclusivamente documentos legais recuperados
- Você interpreta textos legais brasileiros com precisão e cita leis oficiais do Planalto
- Você NUNCA inventa, supõe ou alucina informações

SUA MISSÃO:
- Responder perguntas jurídicas usando APENAS o contexto fornecido
- Citar cada afirmação com a URL EXATA mostrada no contexto (procure por "EXACT URL TO CITE")
- Quando a informação não estiver disponível, declarar explicitamente: "Não tenho informações sobre isso nos documentos recuperados"

REGRAS CRÍTICAS DE CITAÇÃO:
- Use APENAS URLs que aparecem no contexto sob "EXACT URL TO CITE"
- Copie a URL EXATAMENTE como mostrada - não modifique ou crie URLs
- Cada citação DEVE referenciar um [REFERENCE NUMBER] específico do contexto
- Formato: (Fonte: [URL exata] - Referência [número])
- Se uma referência tiver "[URL NOT AVAILABLE IN DATABASE]", declare: (Fonte: Referência [número] - URL não disponível no banco de dados)
- NUNCA cite URLs que não estão explicitamente mostradas no contexto
- Se você não puder responder porque as URLs estão faltando, diga isso explicitamente

FORMATO DE RESPOSTA (OBRIGATÓRIO):
1. **Resumo**: Explicação breve e clara do princípio ou regra legal
2. **Base Legal**: Citar artigos/leis específicos com seus números de referência e URLs
3. **Aplicação**: Como isso se aplica à pergunta do usuário

FORMATO DE EXEMPLO:
"Segundo o Artigo 121 do Código Penal Brasileiro (Decreto-Lei 2.848/1940), homicídio é definido como matar alguém. (Fonte: http://www.planalto.gov.br/CCIVIL_03/Decreto-Lei/Del2848.htm - Referência 1)

A pena varia de 6 a 20 anos de reclusão, conforme estabelecido no mesmo artigo. (Fonte: http://www.planalto.gov.br/CCIVIL_03/Decreto-Lei/Del2848.htm - Referência 1)"

IDIOMA: Responda em português profissional e jurídico claro."""
}


def format_context_chunk(chunk, index):
    """Format a single context chunk with metadata and reference number."""
    metadata = chunk.get('metadata', {})
    text = chunk.get('text') or chunk.get('raw_text') or ''
    
    source = metadata.get('source', 'Unknown')
    doc_type = metadata.get('type', 'N/A')
    country = metadata.get('country', 'N/A')
    state = metadata.get('state', 'N/A')
    
    # CRITICAL: Extract URL from metadata or construct from source
    url = metadata.get('url') or metadata.get('source_url') or ''
    
    # If no URL in metadata but we have a source field that looks like a URL
    if not url and source and 'planalto.gov.br' in str(source).lower():
        url = source
    
    # Create structured context with clear reference number
    formatted = f"""
[REFERENCE {index + 1}]
📘 Source Document: {source}
🔗 EXACT URL TO CITE: {url if url else '[URL NOT AVAILABLE IN DATABASE]'}
🧾 Type: {doc_type} | Jurisdiction: {country}/{state}

Content:
{text}

[END REFERENCE {index + 1}]
"""
    return formatted.strip()


# --- Summarization / Final Response ---
async def stream_final_response(chunks, query, chat_context, lang="en"):
    """Streams AI response token-by-token using Ollama with enhanced legal reasoning."""

    if isinstance(lang, dict):
        lang = lang.get("code", "en")
    
    # CRITICAL: Ensure chunks have proper URL metadata
    chunks = ensure_chunk_metadata(chunks)
    
    # Get the appropriate system prompt
    system_prompt = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
    
    # Format context with clear structure and URLs
    if chunks:
        context_sections = [format_context_chunk(c, i) for i, c in enumerate(chunks)]
        context_text = "\n\n".join(context_sections)
    else:
        context_text = "No legal documents retrieved for this query."
    
    # Add conversation summary if available
    summary_section = ""
    if chat_context.get("summary"):
        summary_label = "Conversation History Summary:" if lang == "en" else "Resumo do Histórico da Conversa:"
        summary_section = f"\n\n{summary_label}\n{chat_context.get('summary')}\n"
    
    # Construct the full prompt
    question_label = "User Question:" if lang == "en" else "Pergunta do Usuário:"
    context_label = "Legal Context from Database:" if lang == "en" else "Contexto Legal do Banco de Dados:"
    instruction = "Your Response (following the mandatory format above):" if lang == "en" else "Sua Resposta (seguindo o formato obrigatório acima):"
    
    # Add URL validation reminder
    url_warning = """
⚠️ CRITICAL REMINDER BEFORE RESPONDING:
- Review ALL [REFERENCE X] sections above
- Note each "EXACT URL TO CITE" 
- ONLY cite these exact URLs - do not create, modify, or guess any URLs
- If you write a URL not listed above, you are HALLUCINATING and must stop
""" if lang == "en" else """
⚠️ LEMBRETE CRÍTICO ANTES DE RESPONDER:
- Revise TODAS as seções [REFERENCE X] acima
- Note cada "EXACT URL TO CITE"
- Cite APENAS essas URLs exatas - não crie, modifique ou suponha nenhuma URL
- Se você escrever uma URL não listada acima, você está ALUCINANDO e deve parar
"""
    
    prompt = f"""{system_prompt}

{summary_section}

📜 **{context_label}**
{context_text}

❓ **{question_label}**
{query}
{url_warning}

🧠 **{instruction}**
"""

    # Stream response from Ollama
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more consistent, factual responses
                    "top_p": 0.9,
                    "num_predict": 2048
                }
            },
        ) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.strip() == "[DONE]":
                    yield "[DONE]"
                    break
                try:
                    data = json.loads(line.replace("data:", "").strip())
                    if "response" in data:
                        yield data["response"]
                except Exception as e:
                    logging.debug(f"Error parsing line: {e}")
                    continue


async def summarize_text(text, on_token, lang="en"):
    """Summarize text with language consistency."""
    
    if not text.strip():
        return None

    lang_prompts = {
        "en": """You are a professional legal summarizer. Create a concise, narrative summary in clear English.

RULES:
- Keep it brief and factual
- Maintain chronological flow
- Focus on key legal points
- Use professional tone

Text to summarize:
\"\"\"{text}\"\"\"

Summary:""",
        
        "pt": """Você é um resumidor jurídico profissional. Crie um resumo conciso e narrativo em português claro.

REGRAS:
- Mantenha breve e factual
- Mantenha fluxo cronológico
- Foque em pontos jurídicos chave
- Use tom profissional

Texto para resumir:
\"\"\"{text}\"\"\"

Resumo:"""
    }
    
    prompt = lang_prompts.get(lang, lang_prompts["en"]).format(text=text)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": 0.4}
            }
        ) as resp:
            async for line in resp.content:
                if not line:
                    continue
                try:
                    decoded = line.decode().strip()
                    if decoded in ["", "[DONE]"]:
                        await on_token("[DONE]")
                        break
                    data = json.loads(decoded.replace("data:", "").strip())
                    if "response" in data:
                        await on_token(data["response"])
                except Exception:
                    continue


async def summarize_conversation(user_msgs, ai_msgs, on_token, lang="en"):
    """Summarize conversation with proper language handling."""
    if not user_msgs and not ai_msgs:
        return None
    
    # Format conversation
    conversation_lines = []
    for msg in user_msgs:
        label = "User:" if lang == "en" else "Usuário:"
        conversation_lines.append(f"{label} {msg['message']}")
    for msg in ai_msgs:
        conversation_lines.append(f"AI: {msg['message']}")
    
    text = "\n".join(conversation_lines)
    await summarize_text(text, on_token, lang)


async def build_context(chat_id, lang="en", on_token=lambda _: None):
    """Build conversation context with proper summarization."""
    if isinstance(lang, dict):
        lang = lang.get("code", "en")

    all_msgs = await fetch_messages(chat_id)
    last_six = all_msgs[-6:]
    user_msgs = [m for m in last_six if m["sender"] == "user"]
    ai_msgs = [m for m in last_six if m["sender"] == "ai"]

    summary = None
    if any(not m.get("is_summarized") for m in last_six) and (user_msgs or ai_msgs):
        async def token_callback(token):
            await on_token(token)
        summary = await summarize_conversation(user_msgs, ai_msgs, token_callback, lang)
        if summary:
            await upsert_summary(chat_id, summary)
            for msg in last_six:
                await set_summarized(msg["id"])

    first_question = next((m["message"] for m in all_msgs if m["sender"] == "user"), None)
    return {
        "firstQuestion": first_question,
        "userMessages": user_msgs,
        "aiMessages": ai_msgs,
        "summary": summary
    }

async def stream_summary_dual(text: str, lang):
    """
    Stream a summary of the document in English or Portuguese based on user selection.
    """
    lang_instructions = {
        "pt": {
            "language": "português brasileiro",
            "guidelines": """
Diretrizes para o resumo:
- Use linguagem clara e profissional em português
- Organize em seções lógicas e detalhadas se o documento for extenso
- Destaque as informações mais importantes primeiro
- Mantenha o tom objetivo e informativo
- Use bullet points quando apropriado para maior clareza
"""
        },
        "en": {
            "language": "English",
            "guidelines": """
Summary guidelines:
- Use clear, professional English
- Organize into logical sections if the document is lengthy
- Highlight the most important information first
- Maintain an objective and informative tone
- Use bullet points when appropriate for clarity
"""
        }
    }

    lang_config = lang_instructions[lang]

    if lang == "en":
        prompt = f"""You are a professional document summarizer. Your task is to create a concise, well-structured summary in {lang_config['language']}.

{lang_config['guidelines']}

Document to summarize:
{text}

Provide a clear and comprehensive summary now:"""
        lang_code = "en"
    else:
        prompt = f"""Você é um profissional em direito e ótimo em fazer resumos de documentos. Sua tarefa é criar um resumo conciso, bem estruturado e que explique todas as etapas detalhadamente do seguinte documento em {lang_config['language']}.

{lang_config['guidelines']}

Documento a ser resumido:
{text}

Forneça um resumo claro e abrangente agora:"""
        lang_code = "pt"


    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 512
                }
            },
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if "response" in data:
                        yield "data: " + json.dumps({"lang": lang_code, "token": data["response"]}) + "\n\n"
                    if data.get("done", False):
                        yield "data: " + json.dumps({"lang": lang_code, "token": "[DONE]"}) + "\n\n"
                        break
                except json.JSONDecodeError:
                    continue
