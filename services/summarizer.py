import aiohttp
from services.chat import fetch_messages, upsert_summary, set_summarized
import json

OLLAMA_URL = "http://localhost:11434"

async def summarize_text(text, on_token, val_lang):
    if not text.strip():
        return None

    # Map short codes to full instructions
    lang_instruction = {
        "en": "Summarize the text in clear, concise English.",
        "pt": "Resuma o texto de forma clara e concisa em Português."
    }.get(val_lang, "Summarize the text clearly.")

    prompt = f"""
You are a professional legal summarizer. {lang_instruction}
- Keep it narrative, avoid lists.
- Make it readable for someone without prior context.
- Highlight important decisions, questions, or info.

Text:
\"\"\"{text}\"\"\"
"""

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": True}
        ) as resp:
            async for line in resp.content:
                if not line:
                    continue
                try:
                    decoded = line.decode().strip()
                    if not decoded or decoded == "[DONE]":
                        await on_token("[DONE]")
                        break
                    json_data = decoded.replace("data:", "").strip()
                    data = json.loads(json_data)
                    if "response" in data:
                        await on_token(data["response"])
                except Exception:
                    continue

async def summarize_conversation(user_msgs, ai_msgs, on_token, lang):
    if not user_msgs and not ai_msgs:
        return None

    text = "\n".join(
        [f"User: {m['message']}" for m in user_msgs] +
        [f"AI: {m['message']}" for m in ai_msgs]
    )
    await summarize_text(text, on_token, lang)

async def build_context(chat_id, lang, on_token=lambda _: None):
    all_msgs = await fetch_messages(chat_id)
    last_six = all_msgs[-6:]
    user_msgs = [m for m in last_six if m["sender"] == "user"]
    ai_msgs = [m for m in last_six if m["sender"] == "ai"]

    needs_summary = any(not m.get("is_summarized") for m in last_six)
    summary = None

    if needs_summary and (user_msgs or ai_msgs):
        async def token_callback(token):
            await on_token(token)
        summary = await summarize_conversation(user_msgs, ai_msgs, token_callback, lang)
        if summary:
            await upsert_summary(chat_id, summary)
            for msg in last_six:
                await set_summarized(msg["id"])

    first_question = next((m["message"] for m in all_msgs if m["sender"] == "user"), None)
    return {"firstQuestion": first_question, "userMessages": user_msgs, "aiMessages": ai_msgs, "summary": summary}
