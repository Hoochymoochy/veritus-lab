from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from services.summarizer import build_context
from services.embedder import embed_and_search, incremental_embed_and_stream

import json

router = APIRouter()

@router.post("/ask")
async def ask(req: Request):
    body = await req.json()
    query = body.get("query")
    chat_id = body.get("id")
    lang = body.get("lang")

    async def event_stream():
        try:
            chat_context = await build_context(chat_id, lang)
            chunks = await embed_and_search(query, "", chat_context)

            print(f"Chunks: {chunks}")

            if not isinstance(chunks, list):
                yield f"data: {json.dumps({'error': 'Bad chunks'})}\n\n"
                return

            async for token in incremental_embed_and_stream(
            [
                f"Source: {c.get('chapter') or c.get('title') or 'Unknown'}\n"
                f"Section: {c.get('section') or 'N/A'}\n"
                f"URL: {c.get('metadata', {}).get('source', 'N/A')}\n\n"
                f"{c.get('text') or c.get('raw_text') or ''}"
                for c in chunks
            ],
                query,
                chat_context,
                lang
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
                if token == "[DONE]":
                    break

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
