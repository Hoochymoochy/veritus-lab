import { createClient } from "@supabase/supabase-js";
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

// get all messages for a chat
export async function fetchMessages(id) {
    const { data, error } = await supabase
        .from("messages")
        .select("*")
        .eq("chat_id", id)
        .order("created_at", { ascending: true });
    if (error) throw error;
    return data;
}

// get the summary for a chat
export async function getSummary(id) {
    const { data, error } = await supabase
        .from("summaries")
        .select("*")
        .eq("chat_id", id)
        .single();

    if (error) throw error;
    return data;
}

// update/insert the summary
export async function upsertSummary(id, content) {
    const { data, error } = await supabase
        .from("summaries")
        .upsert({ chat_id: id, content })
        .select();

    if (error) throw error;
    return data;
}

// set summerized to true
export async function setSummarized(id) {
    const { data, error } = await supabase
        .from("messages")
        .update({ is_summarized: true })
        .eq("id", id)
        .select();
}
