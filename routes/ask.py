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
from services.embeddings import embed_and_search
from services.llm import stream_final_response

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
    logging.info("="*80)
    logging.info("🛰️  NEW REQUEST TO /ask")
    
    body = await req.json()
    query = body.get("query")
    chat_id = body.get("id")
    lang = body.get("lang")
    country = body.get("country")
    state = body.get("state")

    logging.info(f"📋 Request params:")
    logging.info(f"   query: '{query}'")
    logging.info(f"   chat_id: {chat_id}")
    logging.info(f"   lang: {lang}")
    logging.info(f"   country: {country}")
    logging.info(f"   state: {state}")

    async def event_stream():
        try:
            # Step 1: Build conversation context
            logging.info("="*60)
            logging.info("STEP 1: Building conversation context")
            logging.info("="*60)
            chat_context = await build_context(chat_id, lang)
            logging.info(f"✅ Context built | type={type(chat_context)}")
            logging.info(f"   - firstQuestion: {'Yes' if chat_context.get('firstQuestion') else 'No'}")
            logging.info(f"   - userMessages: {len(chat_context.get('userMessages', []))}")
            logging.info(f"   - aiMessages: {len(chat_context.get('aiMessages', []))}")
            logging.info(f"   - summary: {'Yes' if chat_context.get('summary') else 'No'}")

            # Step 2: Search for relevant legal documents
            logging.info("="*60)
            logging.info("STEP 2: Searching for relevant documents")
            logging.info("="*60)
            chunks = await embed_and_search(query, chat_context, country, state)
            
            logging.info(f"✅ Search completed | type={type(chunks)}")
            if isinstance(chunks, list):
                logging.info(f"   Retrieved {len(chunks)} chunks")
                for i, chunk in enumerate(chunks[:3]):  # Log first 3 chunks
                    logging.debug(f"   Chunk {i+1}:")
                    logging.debug(f"      - title: {chunk.get('title', 'N/A')}")
                    logging.debug(f"      - chapter: {chunk.get('chapter', 'N/A')}")
                    logging.debug(f"      - text preview: {str(chunk.get('text', ''))[:100]}...")
            else:
                logging.error(f"❌ embed_and_search returned invalid type: {type(chunks)}")
                yield f"data: {json.dumps({'error': 'Invalid chunks type returned from search'})}\n\n"
                return

            if not chunks:
                logging.warning("⚠️  No chunks returned from search")

            # Step 3: Stream AI response
            logging.info("="*60)
            logging.info("STEP 3: Streaming AI response")
            logging.info("="*60)
            logging.info(f"🚀 Starting stream_final_response with {len(chunks)} chunks...")
            
            token_count = 0
            async for token in stream_final_response(chunks, query, chat_context, lang):
                token_count += 1
                
                if token_count <= 5 or token_count % 100 == 0:
                    logging.debug(f"🪄 Token #{token_count}: '{token[:30]}...'")
                
                yield f"data: {json.dumps({'token': token})}\n\n"
                
                if token == "[DONE]":
                    logging.info(f"✅ Stream finished cleanly | total_tokens={token_count}")
                    break
                
                if token.startswith("[ERROR"):
                    logging.error(f"❌ Error token received: {token}")
                    break
                    
                if await req.is_disconnected():
                    logging.warning("⚠️  Client disconnected mid-stream")
                    break
            
            if token_count == 0:
                logging.error("❌ NO TOKENS WERE YIELDED FROM stream_final_response!")
            else:
                logging.info(f"📊 Stream summary: {token_count} tokens yielded")

        except Exception as e:
            err_trace = traceback.format_exc()
            logging.error("="*60)
            logging.error("💥 EXCEPTION IN /ask EVENT STREAM")
            logging.error("="*60)
            logging.error(f"Exception type: {type(e).__name__}")
            logging.error(f"Exception message: {e}")
            logging.error(f"Full traceback:\n{err_trace}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    logging.info("📡 Returning StreamingResponse...")
    return StreamingResponse(event_stream(), media_type="text/event-stream")