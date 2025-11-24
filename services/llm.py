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
    logging.info(f"🎯 stream_final_response called | chunks={len(chunks) if isinstance(chunks, list) else 'not-list'} | query='{query[:50]}...' | lang={lang}")
    
    if isinstance(lang, dict):
        lang = lang.get("code", "en")
        logging.info(f"📝 Extracted lang code: {lang}")
    
    # CRITICAL: Ensure chunks have proper URL metadata
    chunks = ensure_chunk_metadata(chunks)
    logging.info(f"✅ Chunks metadata ensured | count={len(chunks)}")
    
    # DEBUG: Inspect chunk structure
    logging.info("="*60)
    logging.info("🔍 DEBUGGING CHUNK STRUCTURE:")
    logging.info("="*60)
    if chunks:
        chunk = chunks[0]
        logging.info(f"📦 First chunk keys: {list(chunk.keys())}")
        logging.info(f"📦 Chunk type: {type(chunk)}")
        
        # Check all possible text fields
        text_fields = ['text', 'raw_text', 'content', 'chunk_text', 'body', 'article_text', 'page_content']
        for field in text_fields:
            if field in chunk:
                text_val = chunk[field]
                preview = str(text_val)[:150] if text_val else "EMPTY/NONE"
                logging.info(f"  ✅ '{field}': {preview}...")
        
        # Check metadata structure
        if 'metadata' in chunk:
            meta = chunk['metadata']
            logging.info(f"📦 Metadata keys: {list(meta.keys())}")
            for key, val in meta.items():
                preview = str(val)[:100] if val else "EMPTY/NONE"
                logging.info(f"  📎 metadata['{key}']: {preview}")
        
        # Show full chunk for first one
        logging.info("="*60)
        logging.info(f"📄 FULL FIRST CHUNK:\n{chunk}")
        logging.info("="*60)
    else:
        logging.warning("⚠️  No chunks to inspect!")
    logging.info("="*60)
    
    # Get the appropriate system prompt
    system_prompt = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
    logging.info(f"📋 System prompt retrieved | length={len(system_prompt)} chars")
    
    # Format context with clear structure and URLs
    if chunks:
        context_sections = [format_context_chunk(c, i) for i, c in enumerate(chunks)]
        context_text = "\n\n".join(context_sections)
        logging.info(f"📚 Context formatted | sections={len(context_sections)} | total_chars={len(context_text)}")
    else:
        context_text = "No legal documents retrieved for this query."
        logging.warning("⚠️  No chunks available for context")
    
    # Add conversation summary if available
    summary_section = ""
    if chat_context.get("summary"):
        summary_label = "Conversation History Summary:" if lang == "en" else "Resumo do Histórico da Conversa:"
        summary_section = f"\n\n{summary_label}\n{chat_context.get('summary')}\n"
        logging.info(f"💬 Summary added | length={len(chat_context.get('summary'))} chars")
    else:
        logging.info("💬 No summary in chat context")
    
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
    
    logging.info(f"📝 User message constructed | total_length={len(user_message)} chars")
    
    # DEBUG: Log the full messages being sent to OpenAI
    logging.info("="*60)
    logging.info("📤 MESSAGES BEING SENT TO OPENAI:")
    logging.info("="*60)
    logging.info(f"SYSTEM PROMPT:\n{system_prompt}\n")
    logging.info(f"USER MESSAGE:\n{user_message}\n")
    logging.info("="*60)

    try:
        logging.info(f"🚀 Calling OpenAI API | model={LLM_MODEL} | temp={LLM_TEMPERATURE} | max_tokens={LLM_MAX_TOKENS}")
        
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
        
        logging.info("✅ OpenAI stream created, starting to iterate...")
        token_count = 0
        full_response = []  # Collect full response for debugging
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                full_response.append(token)
                token_count += 1
                if token_count <= 5 or token_count % 50 == 0:
                    logging.debug(f"🪄 Token #{token_count}: '{token[:20]}...'")
                yield token
            
            # Check if stream is done
            if chunk.choices[0].finish_reason == "stop":
                complete_response = "".join(full_response)
                logging.info(f"✅ Stream completed | total_tokens={token_count} | finish_reason=stop")
                logging.info("="*60)
                logging.info("📥 COMPLETE AI RESPONSE:")
                logging.info("="*60)
                logging.info(complete_response)
                logging.info("="*60)
                yield "[DONE]"
                break
        
        logging.info(f"🏁 Stream iteration finished | total_tokens_yielded={token_count}")
                
    except Exception as e:
        logging.error(f"💥 Error in stream_final_response: {e}", exc_info=True)
        yield f"[ERROR: {str(e)}]"


async def summarize_text(text, lang="en"):
    """
    Summarize text and return the complete summary.
    
    Args:
        text: Text to summarize
        lang: Language code ('en' or 'pt')
    
    Returns:
        str: Complete summary text or None
    """
    logging.info(f"📝 summarize_text called | text_length={len(text)} | lang={lang}")
    
    if not text.strip():
        logging.warning("⚠️  Empty text provided, returning None")
        return None

    prompt = SUMMARIZATION_PROMPTS.get(lang, SUMMARIZATION_PROMPTS["en"]).format(text=text)
    logging.info(f"📋 Summarization prompt created | length={len(prompt)} chars")
    logging.debug(f"📄 Prompt preview: {prompt[:200]}...")

    try:
        logging.info(f"🚀 Calling OpenAI for summarization | model={LLM_MODEL}")
        
        # Accumulate the complete response
        summary_parts = []
        
        stream = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=SUMMARY_TEMPERATURE,
            max_tokens=SUMMARY_MAX_TOKENS,
            stream=True
        )
        
        logging.info("✅ Summary stream created, accumulating tokens...")
        token_count = 0
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                summary_parts.append(content)
                token_count += 1
                if token_count <= 3 or token_count % 20 == 0:
                    logging.debug(f"📝 Summary token #{token_count}: '{content[:30]}...'")
            
            if chunk.choices[0].finish_reason == "stop":
                logging.info(f"✅ Summary stream completed | tokens={token_count}")
                break
        
        # Return the complete summary
        complete_summary = "".join(summary_parts)
        logging.info(f"✅ Summary generated | length={len(complete_summary)} chars")
        logging.debug(f"📄 Summary preview: {complete_summary[:200]}...")
        
        return complete_summary
                
    except Exception as e:
        logging.error(f"💥 Error in summarize_text: {e}", exc_info=True)
        return None


async def stream_summary_dual(text: str, lang):
    """
    Stream a summary of the document in English or Portuguese based on user selection.
    
    Args:
        text: Document text to summarize
        lang: Language code ('en' or 'pt')
    
    Yields:
        JSON-formatted tokens with language metadata
    """
    logging.info(f"🌐 stream_summary_dual called | text_length={len(text)} | lang={lang}")
    
    lang_config = DOCUMENT_SUMMARY_INSTRUCTIONS[lang]
    logging.info(f"📋 Language config loaded: {lang_config.get('language')}")

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
    
    logging.info(f"📝 Dual summary prompt created | lang_code={lang_code} | length={len(prompt)}")

    try:
        logging.info("🚀 Starting dual summary stream...")
        
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
        
        logging.info("✅ Dual summary stream created")
        token_count = 0
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                token_count += 1
                if token_count <= 3 or token_count % 30 == 0:
                    logging.debug(f"🪄 Dual summary token #{token_count}")
                yield "data: " + json.dumps({"lang": lang_code, "token": token}) + "\n\n"
            
            if chunk.choices[0].finish_reason == "stop":
                logging.info(f"✅ Dual summary completed | tokens={token_count}")
                yield "data: " + json.dumps({"lang": lang_code, "token": "[DONE]"}) + "\n\n"
                break
                
    except Exception as e:
        logging.error(f"💥 Error in stream_summary_dual: {e}", exc_info=True)
        yield "data: " + json.dumps({"lang": lang_code, "error": str(e)}) + "\n\n"