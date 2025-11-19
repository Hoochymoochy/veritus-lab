from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import logging, json, traceback

from services.extract import extract_text_from_file_bytes, clean_text
from services.ai import stream_summary_dual


router = APIRouter()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@router.post("/summarize-file")
async def summarize_file(
    request: Request,
    file: UploadFile = File(...),
    lang: str = Form(...),
):
    logging.info(f"📥 Received summarize request - File: {file.filename}, Language: {lang}")
    
    # Read file content BEFORE entering the generator
    # This ensures the file is read while still open
    try:
        logging.info(f"📖 Reading file: {file.filename}")
        file_content = await file.read()
        file_size = len(file_content)
        filename = file.filename
        logging.info(f"✅ File read successfully - Size: {file_size} bytes ({file_size / 1024:.2f} KB)")
    except Exception as e:
        logging.error(f"❌ Error reading file: {traceback.format_exc()}")
        async def error_stream():
            yield "data: " + json.dumps({"error": "Failed to read file"}) + "\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    async def event_stream():
        try:
            logging.info(f"🔍 Extracting text from {filename}")
            raw = await extract_text_from_file_bytes(file_content, filename)
            text_length = len(raw)
            logging.info(f"📝 Text extracted - Length: {text_length} characters")
            
            logging.info("🧹 Cleaning extracted text")
            cleaned = clean_text(raw)
            cleaned_length = len(cleaned)
            logging.info(f"✨ Text cleaned - Length: {cleaned_length} characters (reduced by {text_length - cleaned_length})")

            if not cleaned.strip():
                logging.warning("⚠️ File is empty after cleaning")
                yield "data: " + json.dumps({"error": "Empty file"}) + "\n\n"
                return

            logging.info(f"🤖 Starting AI summarization in language: {lang}")
            token_count = 0
            
            async for token in  stream_summary_dual(cleaned, lang):
                if await request.is_disconnected():
                    logging.warning("⚠️ Client disconnected during streaming")
                    break
                token_count += 1
                yield token
            
            logging.info(f"✅ Summarization complete - Streamed {token_count} tokens")

        except Exception as e:
            logging.error(f"❌ Error during summarization: {traceback.format_exc()}")
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")