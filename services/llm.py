"""
LLM interaction service using OpenAI API for generating responses and summaries.
"""
import json
import logging
from openai import AsyncOpenAI
from config import (
    OPENAI_CONFIG,
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

# Initialize OpenAI client
client = AsyncOpenAI(
    api_key=OPENAI_CONFIG["api_key"],
    project=OPENAI_CONFIG.get("project"),
    organization=OPENAI_CONFIG.get("organization")
)


async def stream_final_response(chunks, query, chat_context, lang="en"):
    """
    Streams AI response token-by-token using OpenAI with enhanced legal reasoning.
    
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
    
    # Construct the user message
    question_label = "User Question:" if lang == "en" else "Pergunta do Usuário:"
    context_label = "Legal Context from Database:" if lang == "en" else "Contexto Legal do Banco de Dados:"
    instruction = "Your Response (following the mandatory format above):" if lang == "en" else "Sua Resposta (seguindo o formato obrigatório acima):"
    
    # Add URL validation reminder
    url_warning = URL_VALIDATION_WARNING.get(lang, URL_VALIDATION_WARNING["en"])
    
    user_message = f"""{summary_section}

📜 **{context_label}**
{context_text}

❓ **{question_label}**
{query}
{url_warning}

🧠 **{instruction}**
"""

    try:
        # Stream response from OpenAI
        stream = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            max_tokens=LLM_MAX_TOKENS,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
            
            # Check if stream is done
            if chunk.choices[0].finish_reason == "stop":
                yield "[DONE]"
                break
                
    except Exception as e:
        logging.error(f"Error in stream_final_response: {e}")
        yield f"[ERROR: {str(e)}]"


async def summarize_text(text, on_token, lang="en"):
    """
    Summarize text with language consistency using OpenAI.
    
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

    try:
        stream = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=SUMMARY_TEMPERATURE,
            max_tokens=SUMMARY_MAX_TOKENS,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                await on_token(chunk.choices[0].delta.content)
            
            if chunk.choices[0].finish_reason == "stop":
                await on_token("[DONE]")
                break
                
    except Exception as e:
        logging.error(f"Error in summarize_text: {e}")
        await on_token(f"[ERROR: {str(e)}]")


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

    try:
        stream = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            max_tokens=SUMMARY_MAX_TOKENS,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                yield "data: " + json.dumps({"lang": lang_code, "token": token}) + "\n\n"
            
            if chunk.choices[0].finish_reason == "stop":
                yield "data: " + json.dumps({"lang": lang_code, "token": "[DONE]"}) + "\n\n"
                break
                
    except Exception as e:
        logging.error(f"Error in stream_summary_dual: {e}")
        yield "data: " + json.dumps({"lang": lang_code, "error": str(e)}) + "\n\n"