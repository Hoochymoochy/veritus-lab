"""
Conversation management and summarization service.
"""
import logging
from services.chat import fetch_messages, upsert_summary, set_summarized
from services.llm import summarize_text


async def summarize_conversation(user_msgs, ai_msgs, lang="en"):
    """
    Summarize conversation and return the summary text.
    
    Args:
        user_msgs: List of user messages
        ai_msgs: List of AI messages
        lang: Language code ('en' or 'pt')
    
    Returns:
        str: Conversation summary or None
    """
    logging.info(f"💬 summarize_conversation called | user_msgs={len(user_msgs)} | ai_msgs={len(ai_msgs)} | lang={lang}")
    
    if not user_msgs and not ai_msgs:
        logging.warning("⚠️  No messages to summarize, returning None")
        return None
    
    # Format conversation
    conversation_lines = []
    for i, msg in enumerate(user_msgs):
        label = "User:" if lang == "en" else "Usuário:"
        msg_text = msg.get('message', '')
        conversation_lines.append(f"{label} {msg_text}")
        logging.debug(f"  📝 User msg {i+1}: {msg_text[:50]}...")
    
    for i, msg in enumerate(ai_msgs):
        msg_text = msg.get('message', '')
        conversation_lines.append(f"AI: {msg_text}")
        logging.debug(f"  🤖 AI msg {i+1}: {msg_text[:50]}...")
    
    text = "\n".join(conversation_lines)
    logging.info(f"📄 Conversation formatted | total_length={len(text)} chars | lines={len(conversation_lines)}")
    
    logging.info("🚀 Calling summarize_text...")
    summary = await summarize_text(text, lang)
    
    if summary:
        logging.info(f"✅ Summary received | length={len(summary)} chars")
        logging.debug(f"📄 Summary preview: {summary[:150]}...")
    else:
        logging.warning("⚠️  summarize_text returned None or empty")
    
    return summary


async def build_context(chat_id, lang="en"):
    """
    Build conversation context with proper summarization.
    
    Args:
        chat_id: ID of the chat conversation
        lang: Language code ('en' or 'pt')
    
    Returns:
        Dictionary containing conversation context including:
        - firstQuestion: First user question in the conversation
        - userMessages: Last 6 user messages
        - aiMessages: Last 6 AI messages
        - summary: Conversation summary if available
    """
    logging.info(f"🏗️  build_context called | chat_id={chat_id} | lang={lang}")
    
    if isinstance(lang, dict):
        lang = lang.get("code", "en")
        logging.info(f"📝 Extracted lang from dict: {lang}")

    logging.info("📥 Fetching messages from database...")
    all_msgs = await fetch_messages(chat_id)
    logging.info(f"✅ Fetched {len(all_msgs) if all_msgs else 0} total messages")

    last_six = all_msgs[-6:] if all_msgs else []
    logging.info(f"📊 Processing last {len(last_six)} messages")
    
    user_msgs = [m for m in last_six if m.get("sender") == "user"]
    ai_msgs = [m for m in last_six if m.get("sender") == "ai"]
    
    logging.info(f"👤 User messages in last 6: {len(user_msgs)}")
    logging.info(f"🤖 AI messages in last 6: {len(ai_msgs)}")
    
    for i, msg in enumerate(user_msgs):
        logging.debug(f"  User msg {i+1}: id={msg.get('id')} | summarized={msg.get('is_summarized')} | text='{msg.get('message', '')[:50]}...'")
    
    for i, msg in enumerate(ai_msgs):
        logging.debug(f"  AI msg {i+1}: id={msg.get('id')} | summarized={msg.get('is_summarized')} | text='{msg.get('message', '')[:50]}...'")

    summary = None
    try:
        # Check if any messages need summarization
        needs_summary = any(not m.get("is_summarized") for m in last_six)
        has_messages = bool(user_msgs or ai_msgs)
        
        logging.info(f"🔍 Summarization check | needs_summary={needs_summary} | has_messages={has_messages}")
        
        if needs_summary and has_messages:
            logging.info("🚀 Starting conversation summarization...")
            summary = await summarize_conversation(user_msgs, ai_msgs, lang)
            
            if summary:
                logging.info(f"✅ Summary generated successfully | length={len(summary)} chars")
                logging.info("💾 Upserting summary to database...")
                await upsert_summary(chat_id, summary)
                logging.info("✅ Summary saved to database")
                
                # Mark messages as summarized
                logging.info(f"🏷️  Marking {len(last_six)} messages as summarized...")
                for msg in last_six:
                    msg_id = msg.get("id")
                    if msg_id:
                        await set_summarized(msg_id)
                        logging.debug(f"  ✅ Marked message {msg_id} as summarized")
                logging.info("✅ All messages marked as summarized")
            else:
                logging.warning("⚠️  Summary generation returned None")
        else:
            logging.info("ℹ️  No summarization needed (all messages already summarized or no messages)")
            
    except Exception as e:
        logging.error(f"💥 Error summarizing conversation for chat {chat_id}: {e}", exc_info=True)

    # Get first question
    first_question = next((m.get("message") for m in all_msgs if m.get("sender") == "user"), None)
    if first_question:
        logging.info(f"❓ First question found: '{first_question[:50]}...'")
    else:
        logging.warning("⚠️  No first question found")
    
    context = {
        "firstQuestion": first_question,
        "userMessages": user_msgs,
        "aiMessages": ai_msgs,
        "summary": summary
    }
    
    logging.info(f"✅ Context built | first_question={'Yes' if first_question else 'No'} | summary={'Yes' if summary else 'No'}")
    
    return context