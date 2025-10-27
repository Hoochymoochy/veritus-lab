from utils.embeder import model
from utils.pinecode import insert_pinecone_index
import uuid

async def embed_text(text: str):
    embedding = model.encode([text])[0].tolist()
    return embedding

async def embed_and_store(text, metadata=None, namespace=""):
    vector = await embed_text(text)
    insert_pinecone_index.upsert([(str(uuid.uuid4()), vector, {**metadata, "text": text})], namespace=namespace)
    return {"id": str(uuid.uuid4()), "success": True}

async def embed_and_search(query, namespace="", context=None):
    from utils.pinecode import search_legal_docs
    return search_legal_docs(query)

# incremental_embed_and_stream.py
from services.final_response import stream_final_response

async def incremental_embed_and_stream(texts, query, chat_context, lang):
    chunks = []
    
    for text in texts:
        chunks.append({"text": text, "vector": await embed_text(text)})
    
    # Stream the final response with all chunks
    async for token in stream_final_response(chunks, query, chat_context, lang):
        yield token

