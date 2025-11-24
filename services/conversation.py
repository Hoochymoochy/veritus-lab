"""
Conversation management and summarization service.
"""
from services.chat import fetch_messages, upsert_summary, set_summarized
from services.llm import summarize_text


async def summarize_conversation(user_msgs, ai_msgs, on_token, lang="en"):
    """
    Summarize conversation with proper language handling.
    
    Args:
        user_msgs: List of user messages
        ai_msgs: List of AI messages
        on_token: Callback function for tokens
        lang: Language code ('en' or 'pt')
    
    Returns:
        None (streams via on_token callback)
    """
    if not user_msgs and not ai_msgs:
        return None
    
    # Format conversation
    conversation_lines = []
    for msg in user_msgs:
        label = "User:" if lang == "en" else "Usuário:"
        conversation_lines.append(f"{label} {msg['message']}")
    for msg in ai_msgs:
        conversation_lines.append(f"AI: {msg['message']}")
    
    text = "\n".join(conversation_lines)
    await summarize_text(text, on_token, lang)


async def build_context(chat_id, lang="en", on_token=lambda _: None):
    """
    Build conversation context with proper summarization.
    
    Args:
        chat_id: ID of the chat conversation
        lang: Language code ('en' or 'pt')
        on_token: Callback function for streaming tokens
    
    Returns:
        Dictionary containing conversation context including:
        - firstQuestion: First user question in the conversation
        - userMessages: Last 6 user messages
        - aiMessages: Last 6 AI messages
        - summary: Conversation summary if available
    """
    if isinstance(lang, dict):
        lang = lang.get("code", "en")

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
    return {
        "firstQuestion": first_question,
        "userMessages": user_msgs,
        "aiMessages": ai_msgs,
        "summary": summary
    }