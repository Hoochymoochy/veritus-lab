# final_response.py
import asyncio
import httpx
import os
import json

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


async def stream_final_response(chunks, query, chat_context, lang="en"):
    """
    Streams AI response token-by-token with bilingual support.
    lang: "en" for English, "pt" for Portuguese
    """
    context_text = "\n".join(c.get("text") or c.get("raw_text") or "" for c in chunks)
    summary_text = f"Summary:\n{chat_context.get('summary')}\n" if chat_context.get("summary") else ""

    # Map lang to instruction
    lang_instruction = {
        "en": "Answer the question in clear, concise English.",
        "pt": "Responda à pergunta de forma clara e concisa em Português."
    }.get(lang, "Answer clearly.")

    prompt = f"""
You are a professional legal AI assistant. {lang_instruction}

- Use the context below to answer the user's question clearly and concisely.
- If the context includes a URL, include it at the end of your answer in Markdown link format like: [View Law Here](https://example.com)
- Do NOT make up URLs. Only use those provided in context.
- Keep your response structured and human-readable.
- Highlight important legal details, articles, or clauses when relevant.

Context:
{summary_text}{context_text}

User Question: {query}

Answer in a narrative style.
Stream token by token.
"""

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", f"{OLLAMA_URL}/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": True}
        ) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.strip() == "[DONE]":
                    yield "[DONE]"
                    break

                # Clean up Ollama's `data:` lines
                clean = line.replace("data:", "").strip()
                try:
                    data = json.loads(clean)
                except Exception:
                    continue

                if "response" in data:
                    yield data["response"]
