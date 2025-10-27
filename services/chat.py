from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

async def fetch_messages(chat_id):
    try:
        res = supabase.table("messages").select("*").eq("chat_id", chat_id).order("created_at", desc=False).execute()
        return res.data
    except Exception as e:
        raise Exception(f"Supabase error: {str(e)}")

async def get_summary(chat_id):
    try:
        res = supabase.table("summaries").select("*").eq("chat_id", chat_id).single().execute()
        return res.data
    except Exception as e:
        raise Exception(f"Supabase error: {str(e)}")

async def upsert_summary(chat_id, content):
    try:
        res = supabase.table("summaries").upsert({"chat_id": chat_id, "content": content}).execute()
        return res.data
    except Exception as e:
        raise Exception(f"Supabase error: {str(e)}")

async def set_summarized(msg_id):
    try:
        res = supabase.table("messages").update({"is_summarized": True}).eq("id", msg_id).execute()
        return res.data
    except Exception as e:
        raise Exception(f"Supabase error: {str(e)}")