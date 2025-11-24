"""
Embedding generation and search functionality.
"""
from sentence_transformers import SentenceTransformer
from utils.pinecode import search_legal_docs
from config import EMBEDDING_MODEL

# Initialize embedding model
model = SentenceTransformer(EMBEDDING_MODEL)


async def embed_text(text: str):
    """Generate embedding vector for a single text."""
    return model.encode([text])[0].tolist()


async def embed_and_search(query, context=None, country=None, state=None):
    """
    Embed query and search legal documents.
    
    Args:
        query: Search query string
        context: Optional context information
        country: Optional country filter
        state: Optional state filter
    
    Returns:
        Search results from legal documents database
    """
    return search_legal_docs(query, context=context, country=country, state=state)


async def incremental_embed_and_stream(texts, query, chat_context, lang="en"):
    """
    Generate embeddings for text chunks and stream final AI response.
    
    Args:
        texts: List of text strings to embed
        query: User's query
        chat_context: Conversation context
        lang: Language code ('en' or 'pt')
    
    Yields:
        Tokens from the streaming response
    """
    from services.llm import stream_final_response
    
    chunks = [{"text": t, "vector": await embed_text(t)} for t in texts]
    async for token in stream_final_response(chunks, query, chat_context, lang):
        yield token