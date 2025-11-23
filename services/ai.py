# ai_openai_integration.py
import os
import json
import logging
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Callable, Optional

from utils.pinecode import search_legal_docs
from services.chat import fetch_messages, upsert_summary, set_summarized
from utils.openAPI import openai

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Configuration ---
EMBED_MODEL = "text-embedding-3-small"  # cheap, multilingual
SUMMARY_MODEL = "gpt-4o-mini"  # low-cost summarizer


# --- Embedding / Search ---
async def embed_text(text: str) -> List[float]:
    """
    Return a single embedding vector for `text` using OpenAI embeddings.
    """
    if not text:
        return []
    res = await openai.embeddings.create(model=EMBED_MODEL, input=text)
    embedding = res.data[0].embedding
    return embedding


async def embed_and_search(query: str, context=None, country=None, state=None):
    """
    Search legal documents and normalize to standard format.
    Returns a list of chunk dictionaries with 'text' and 'metadata' fields.
    """
    # Get results from your existing search
    raw_results = search_legal_docs(query, context=context, country=country, state=state)
    
    # Normalize the results to expected format
    normalized_chunks = []
    
    # Handle different result formats
    if isinstance(raw_results, dict):
        # Case 1: Results wrapped in a dict like {"matches": [...]}
        if "matches" in raw_results:
            items = raw_results["matches"]
        elif "results" in raw_results:
            items = raw_results["results"]
        else:
            # Unknown dict structure, log and return empty
            logging.warning(f"⚠️ Unexpected dict structure: {raw_results.keys()}")
            return []
    elif isinstance(raw_results, list):
        # Case 2: Direct list of results
        items = raw_results
    else:
        logging.warning(f"⚠️ Unexpected result type: {type(raw_results)}")
        return []
    
    # Normalize each item to standard format
    for item in items:
        if not isinstance(item, dict):
            continue
            
        # Extract text content (handle various field names)
        text = (
            item.get("text") or 
            item.get("content") or 
            item.get("raw_text") or 
            item.get("page_content") or
            ""
        )
        
        # Extract or build metadata
        metadata = item.get("metadata", {})
        if not metadata:
            # Build metadata from top-level fields if not nested
            metadata = {
                "source": item.get("source", "Unknown"),
                "url": item.get("url", ""),
                "type": item.get("type", "N/A"),
                "country": item.get("country", "N/A"),
                "state": item.get("state", "N/A"),
            }
        
        normalized_chunks.append({
            "text": text,
            "metadata": metadata,
            "raw_item": item  # Keep original for debugging if needed
        })
    
    logging.info(f"✅ Normalized {len(normalized_chunks)} search results")
    return normalized_chunks


async def incremental_embed_and_stream(texts: List[str], query: str, chat_context: Dict[str, Any], lang="en") -> AsyncGenerator[str, None]:
    """
    Generate embeddings for chunks and stream final AI response.
    Yields strings (tokens or small chunks) to mimic streaming behaviour.
    """
    chunks = [{"text": t, "vector": await embed_text(t)} for t in texts]
    async for token in stream_final_response(chunks, query, chat_context, lang):
        yield token


# --- Helper Functions ---
def ensure_chunk_metadata(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure chunks have proper URL metadata.
    """
    processed_chunks = []
    for chunk in chunks:
        processed = dict(chunk)
        if "metadata" not in processed:
            processed["metadata"] = {}
        metadata = processed["metadata"]

        url = (
            metadata.get("url")
            or metadata.get("source_url")
            or metadata.get("source")
            or chunk.get("source")
            or chunk.get("url")
            or ""
        )

        if url and isinstance(url, str) and "planalto.gov.br" in url.lower():
            metadata["url"] = url
        else:
            metadata["url"] = ""

        processed_chunks.append(processed)

    return processed_chunks


def format_context_chunk(chunk: Dict[str, Any], index: int) -> str:
    """
    Format a single chunk for display in the prompt.
    """
    metadata = chunk.get("metadata", {})
    text = chunk.get("text") or chunk.get("raw_text") or ""
    source = metadata.get("source", "Unknown")
    doc_type = metadata.get("type", "N/A")
    country = metadata.get("country", "N/A")
    state = metadata.get("state", "N/A")
    url = metadata.get("url") or metadata.get("source_url") or ""

    if not url and source and "planalto.gov.br" in str(source).lower():
        url = source

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


# --- System Prompts ---
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

IDIOMA: Responda em português profissional e jurídico claro."""
}


# --- Main Response Generation ---
async def stream_final_response(chunks: List[Dict[str, Any]], query: str, chat_context: Dict[str, Any], lang="en") -> AsyncGenerator[str, None]:
    """
    Build the full prompt, call OpenAI to generate the response,
    then stream the resulting text back in small chunks to mimic token streaming.
    """
    if isinstance(lang, dict):
        lang = lang.get("code", "en")

    chunks = ensure_chunk_metadata(chunks)
    system_prompt = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])

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

    question_label = "User Question:" if lang == "en" else "Pergunta do Usuário:"
    context_label = "Legal Context from Database:" if lang == "en" else "Contexto Legal do Banco de Dados:"
    instruction = "Your Response (following the mandatory format above):" if lang == "en" else "Sua Resposta (seguindo o formato obrigatório acima):"

    url_warning = (
        "\n⚠️ CRITICAL REMINDER BEFORE RESPONDING:\n- Review ALL [REFERENCE X] sections above\n- Note each \"EXACT URL TO CITE\"\n- ONLY cite these exact URLs - do not create, modify, or guess any URLs\n- If you write a URL not listed above, you are HALLUCINATING and must stop\n"
        if lang == "en"
        else "\n⚠️ LEMBRETE CRÍTICO ANTES DE RESPONDER:\n- Revise TODAS as seções [REFERENCE X] acima\n- Note cada \"EXACT URL TO CITE\"\n- Cite APENAS essas URLs exatas - não crie, modifique ou suponha nenhuma URL\n- Se você escrever uma URL não listada acima, você está ALUCINANDO e deve parar\n"
    )

    # Call OpenAI chat completion
    logging.debug("Calling OpenAI for final response...")
    resp = await openai.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{summary_section}\n\n{context_label}\n\n{context_text}\n\n{question_label}\n{query}\n{url_warning}\n\n{instruction}"}
        ],
        max_tokens=1500,
        temperature=0.2,
    )

    # FIXED: Correct way to access the response
    text = resp.choices[0].message.content or ""
    
    # Simple streaming emulation: yield in chunks of ~100 chars
    chunk_size = 100
    for i in range(0, len(text), chunk_size):
        await asyncio.sleep(0)  # yield control
        yield text[i : i + chunk_size]

    yield "[DONE]"


# --- Summarization Functions ---
async def summarize_text(text: str, on_token: Callable[[str], Any], lang="en") -> Optional[str]:
    """
    Summarize text with OpenAI and call on_token for each chunk (to mimic streaming).
    Returns final summary string.
    """
    if not text or not text.strip():
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

Resumo:""",
    }

    prompt_template = lang_prompts.get(lang, lang_prompts["en"])
    prompt = prompt_template.format(text=text)

    resp = await openai.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.3,
    )

    # FIXED: Correct way to access the response
    summary_text = resp.choices[0].message.content or ""

    # Stream out via on_token in small chunks
    chunk_size = 120
    for i in range(0, len(summary_text), chunk_size):
        chunk = summary_text[i : i + chunk_size]
        await on_token(chunk)
        await asyncio.sleep(0)

    # Final token marker
    await on_token("[DONE]")
    return summary_text


async def summarize_conversation(user_msgs: List[Dict[str, Any]], ai_msgs: List[Dict[str, Any]], on_token: Callable[[str], Any], lang="en"):
    """
    Summarize a conversation between user and AI.
    """
    if not user_msgs and not ai_msgs:
        return None

    conversation_lines = []
    for msg in user_msgs:
        label = "User:" if lang == "en" else "Usuário:"
        conversation_lines.append(f"{label} {msg['message']}")
    for msg in ai_msgs:
        conversation_lines.append(f"AI: {msg['message']}")

    text = "\n".join(conversation_lines)
    return await summarize_text(text, on_token, lang)


async def build_context(chat_id, lang="en", on_token=None):
    """
    Build conversation context with optional token streaming callback.
    
    Args:
        chat_id: The chat session ID
        lang: Language code ("en" or "pt")
        on_token: Optional async callback function for streaming tokens
    """
    if isinstance(lang, dict):
        lang = lang.get("code", "en")

    all_msgs = await fetch_messages(chat_id)
    last_six = all_msgs[-6:]
    user_msgs = [m for m in last_six if m["sender"] == "user"]
    ai_msgs = [m for m in last_six if m["sender"] == "ai"]

    summary = None
    if any(not m.get("is_summarized") for m in last_six) and (user_msgs or ai_msgs):
        # Create proper async token callback
        async def token_callback(token):
            if on_token is not None:
                # Check if on_token is async or sync
                if asyncio.iscoroutinefunction(on_token):
                    await on_token(token)
                else:
                    on_token(token)
        
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
        "summary": summary,
    }


async def stream_summary_dual(text: str, lang):
    """
    Stream a summary of the document in English or Portuguese based on user selection.
    Yields Server-Sent-Event style payload strings.
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
        },
    }

    lang_config = lang_instructions.get(lang, lang_instructions["en"])
    lang_code = "en" if lang == "en" else "pt"

    if lang == "en":
        prompt = f"""You are a professional document summarizer. Your task is to create a concise, well-structured summary in {lang_config['language']}.

{lang_config['guidelines']}

Document to summarize:
{text}

Provide a clear and comprehensive summary now:"""
    else:
        prompt = f"""Você é um profissional em direito e ótimo em fazer resumos de documentos. Sua tarefa é criar um resumo conciso, bem estruturado e que explique todas as etapas detalhadamente do seguinte documento em {lang_config['language']}.

{lang_config['guidelines']}

Documento a ser resumido:
{text}

Forneça um resumo claro e abrangente agora:"""

    resp = await openai.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.3,
    )

    # FIXED: Correct way to access the response
    summary_text = resp.choices[0].message.content or ""

    # Yield SSE-like lines
    chunk_size = 120
    for i in range(0, len(summary_text), chunk_size):
        chunk = summary_text[i : i + chunk_size]
        payload = {"lang": lang_code, "token": chunk}
        yield "data: " + json.dumps(payload) + "\n\n"
        await asyncio.sleep(0)
    
    # Final done
    yield "data: " + json.dumps({"lang": lang_code, "token": "[DONE]"}) + "\n\n"