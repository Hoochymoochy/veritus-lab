"""
/ask endpoint - Main chat query route with legal document search
"""
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import logging
import json
import traceback

# Updated imports to use new modular structure
from services.conversation import build_context
from services.embeddings import embed_and_search, incremental_embed_and_stream

router = APIRouter()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@router.post("/ask")
async def ask(req: Request):
    """
    Main endpoint for asking legal questions.
    
    Request body:
        - query: User's question
        - id: Chat ID for conversation context
        - lang: Language code ('en' or 'pt')
        - country: Country filter for search
        - state: State filter for search
    
    Returns:
        StreamingResponse with AI-generated legal answers
    """
    body = await req.json()
    query = body.get("query")
    chat_id = body.get("id")
    lang = body.get("lang")
    country = body.get("country")
    state = body.get("state")

    logging.info(f"🛰️  /ask hit | query='{query}' | chat_id={chat_id} | lang={lang} | country={country} | state={state}")

    async def event_stream():
        try:
            # Step 1: Build conversation context
            logging.info("⚙️  Building chat context...")
            chat_context = await build_context(chat_id, lang)
            logging.info(f"✅ Context ready: {type(chat_context)}")

            # Step 2: Search for relevant legal documents
            logging.info("🔍 Running embed_and_search...")
            chunks = await embed_and_search(query, chat_context, country, state)
            print(f"✅ Retrieved {len(chunks) if isinstance(chunks, list) else 'non-list'} chunks")

            if not isinstance(chunks, list):
                logging.error("❌ embed_and_search returned invalid type")
                yield f"data: {json.dumps({'error': 'Bad chunks'})}\n\n"
                return

            # Step 3: Format chunks for the LLM
            # Note: The chunk processing is now handled internally by the service,
            # but we keep this formatting for backward compatibility
            formatted_chunks = [
                f"Source: {c.get('chapter') or c.get('title') or 'Unknown'}\n"
                f"Section: {c.get('section') or 'N/A'}\n"
                f"URL: {c.get('metadata', {}).get('source', 'N/A')}\n\n"
                f"{c.get('text') or c.get('raw_text') or ''}"
                for c in chunks
            ]
            logging.info(f"🧩 Prepared {len(formatted_chunks)} formatted chunks")

            # Step 4: Stream AI response with embedded chunks
            logging.info("🚀 Starting incremental_embed_and_stream...")
            async for token in incremental_embed_and_stream(formatted_chunks, query, chat_context, lang):
                logging.debug(f"🪄 Token: {token}")
                yield f"data: {json.dumps({'token': token})}\n\n"
                
                if token == "[DONE]":
                    logging.info("✅ Stream finished cleanly")
                    break
                    
                if await req.is_disconnected():
                    logging.info("❌ Client disconnected")
                    break

        except Exception as e:
            err_trace = traceback.format_exc()
            logging.error(f"💥 Exception in /ask: {e}\n{err_trace}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    logging.info("📡 Sending StreamingResponse...")
    return StreamingResponse(event_stream(), media_type="text/event-stream")