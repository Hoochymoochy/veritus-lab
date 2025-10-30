import json, logging, os
import aiohttp, httpx
from sentence_transformers import SentenceTransformer
from utils.pinecode import search_legal_docs
from services.chat import fetch_messages, upsert_summary, set_summarized

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
model = SentenceTransformer("intfloat/multilingual-e5-large")


# --- Embedding / Search ---
async def embed_text(text: str):
    return model.encode([text])[0].tolist()


async def embed_and_search(query, context=None, country=None, state=None):
    return search_legal_docs(query, context=context, country=country, state=state)


async def incremental_embed_and_stream(texts, query, chat_context, lang="en"):
    """Generate embeddings for chunks and stream final AI response."""
    chunks = [{"text": t, "vector": await embed_text(t)} for t in texts]
    async for token in stream_final_response(chunks, query, chat_context, lang):
        yield token


# --- Summarization / Final Response ---
async def stream_final_response(chunks, query, chat_context, lang="en"):
    """Streams AI response token-by-token using Ollama."""
    context_text = "\n".join(c.get("text") or c.get("raw_text") or "" for c in chunks)
    summary_text = f"Summary:\n{chat_context.get('summary')}\n" if chat_context.get("summary") else ""

    lang_instruction = {
        "en": "Answer the question in clear, concise English.",
        "pt": "Responda à pergunta de forma clara e concisa em Português."
    }.get(lang, "Answer clearly.")

    prompt = f"""
You are a professional legal AI assistant. {lang_instruction}
- Use the context below to answer clearly.
- Include URLs if present in context.
- Keep human-readable.

Context:
{summary_text}{context_text}

User Question: {query}

Stream token by token.
"""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json={"model": "mistral", "prompt": prompt, "stream": True}) as resp:
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
                except Exception:
                    continue


async def summarize_text(text, on_token, lang="en"):
    if not text.strip(): return None

    lang_instruction = {
        "en": "Summarize the text in clear, concise English.",
        "pt": "Resuma o texto de forma clara e concisa em Português."
    }.get(lang, "Summarize clearly.")

    prompt = f"""
You are a professional legal summarizer. {lang_instruction}
- Keep it narrative.
Text:
\"\"\"{text}\"\"\"
"""
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{OLLAMA_URL}/api/generate", json={"model": "mistral", "prompt": prompt, "stream": True}) as resp:
            async for line in resp.content:
                if not line: continue
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


async def summarize_conversation(user_msgs, ai_msgs, on_token, lang="en"):
    if not user_msgs and not ai_msgs: return None
    text = "\n".join([f"User: {m['message']}" for m in user_msgs] + [f"AI: {m['message']}" for m in ai_msgs])
    await summarize_text(text, on_token, lang)


async def build_context(chat_id, lang="en", on_token=lambda _: None):
    all_msgs = await fetch_messages(chat_id)
    last_six = all_msgs[-6:]
    user_msgs = [m for m in last_six if m["sender"] == "user"]
    ai_msgs = [m for m in last_six if m["sender"] == "ai"]

    summary = None
    if any(not m.get("is_summarized") for m in last_six) and (user_msgs or ai_msgs):
        async def token_callback(token):
            await on_token(token)
        summary = await summarize_conversation(user_msgs, ai_msgs, token_callback, lang)
        if summary:
            await upsert_summary(chat_id, summary)
            for msg in last_six:
                await set_summarized(msg["id"])

    first_question = next((m["message"] for m in all_msgs if m["sender"] == "user"), None)
    return {"firstQuestion": first_question, "userMessages": user_msgs, "aiMessages": ai_msgs, "summary": summary}
