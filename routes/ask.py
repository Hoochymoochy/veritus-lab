from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import logging, json, traceback

from services.ai import build_context, embed_and_search, incremental_embed_and_stream

router = APIRouter()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@router.post("/ask")
async def ask(req: Request):
    body = await req.json()
    query = body.get("query")
    chat_id = body.get("id")
    lang = body.get("lang")
    country = body.get("country")
    state = body.get("state")

    logging.info(f"🛰️  /ask hit | query='{query}' | chat_id={chat_id} | lang={lang} | country={country} | state={state}")

    async def event_stream():
        try:
            logging.info("⚙️  Building chat context...")
            chat_context = await build_context(chat_id, lang)
            logging.info(f"✅ Context ready: {type(chat_context)}")

            logging.info("🔍 Running embed_and_search...")
            chunks = await embed_and_search(query, chat_context, country, state)
            logging.info(f"✅ Retrieved {len(chunks) if isinstance(chunks, list) else 'non-list'} chunks")

            if not isinstance(chunks, list):
                logging.error("❌ embed_and_search returned invalid type")
                yield f"data: {json.dumps({'error': 'Bad chunks'})}\n\n"
                return

            # Format each chunk
            formatted_chunks = [
                f"Source: {c.get('chapter') or c.get('title') or 'Unknown'}\n"
                f"Section: {c.get('section') or 'N/A'}\n"
                f"URL: {c.get('metadata', {}).get('source', 'N/A')}\n\n"
                f"{c.get('text') or c.get('raw_text') or ''}"
                for c in chunks
            ]
            logging.info(f"🧩 Prepared {len(formatted_chunks)} formatted chunks")

            logging.info("🚀 Starting incremental_embed_and_stream...")
            async for token in incremental_embed_and_stream(formatted_chunks, query, chat_context, lang):
                logging.debug(f"🪄 Token: {token}")
                yield f"data: {json.dumps({'token': token})}\n\n"
                if token == "[DONE]":
                    logging.info("✅ Stream finished cleanly")
                    break

        except Exception as e:
            err_trace = traceback.format_exc()
            logging.error(f"💥 Exception in /ask: {e}\n{err_trace}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    logging.info("📡 Sending StreamingResponse...")
    return StreamingResponse(event_stream(), media_type="text/event-stream")
