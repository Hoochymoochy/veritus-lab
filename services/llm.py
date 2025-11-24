"""
LLM interaction service for generating responses and summaries.
"""
import json
import logging
import httpx
import aiohttp
from config import (
    OLLAMA_URL, 
    LLM_MODEL, 
    LLM_TEMPERATURE, 
    LLM_TOP_P, 
    LLM_MAX_TOKENS,
    SUMMARY_TEMPERATURE,
    SUMMARY_MAX_TOKENS
)
from prompts import (
    SYSTEM_PROMPTS, 
    SUMMARIZATION_PROMPTS,
    DOCUMENT_SUMMARY_INSTRUCTIONS,
    URL_VALIDATION_WARNING
)
from utils.chunk_processing import ensure_chunk_metadata, format_context_chunk


async def stream_final_response(chunks, query, chat_context, lang="en"):
    """
    Streams AI response token-by-token using Ollama with enhanced legal reasoning.
    
    Args:
        chunks: List of document chunks with metadata
        query: User's query
        chat_context: Conversation context including summary
        lang: Language code ('en' or 'pt')
    
    Yields:
        Response tokens from the LLM
    """
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
    url_warning = URL_VALIDATION_WARNING.get(lang, URL_VALIDATION_WARNING["en"])
    
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
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                    "top_p": LLM_TOP_P,
                    "num_predict": LLM_MAX_TOKENS
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
    """
    Summarize text with language consistency.
    
    Args:
        text: Text to summarize
        on_token: Callback function for each token
        lang: Language code ('en' or 'pt')
    
    Returns:
        None (streams via on_token callback)
    """
    if not text.strip():
        return None

    prompt = SUMMARIZATION_PROMPTS.get(lang, SUMMARIZATION_PROMPTS["en"]).format(text=text)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": SUMMARY_TEMPERATURE}
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


async def stream_summary_dual(text: str, lang):
    """
    Stream a summary of the document in English or Portuguese based on user selection.
    
    Args:
        text: Document text to summarize
        lang: Language code ('en' or 'pt')
    
    Yields:
        JSON-formatted tokens with language metadata
    """
    lang_config = DOCUMENT_SUMMARY_INSTRUCTIONS[lang]

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
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                    "top_p": LLM_TOP_P,
                    "num_predict": SUMMARY_MAX_TOKENS
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