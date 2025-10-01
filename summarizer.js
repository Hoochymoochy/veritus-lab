// 📝 Batch Summarizer Module — Veritus style
import axios from "axios";
import { fetchMessages, upsertSummary, setSummarized } from "./chat.js";

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";

// Single section summarizer
async function summarizeText(text) {
  if (!text || text.trim() === "") return null; // handle empty text gracefully

  const prompt = `
You are a professional legal summarizer. Summarize the following conversation or legal text into a concise, clear, narrative-style summary.
- Focus on clarity and key points.
- Avoid creating a list; make it flow naturally.
- Keep it readable for someone without prior context.
- Highlight decisions, questions, or important information.

Text to summarize:
"""${text}"""
  `;

  const response = await axios.post(`${OLLAMA_URL}/api/generate`, {
    model: "mistral",
    prompt,
    stream: false,
  });

  return response.data.response || null;
}

// Summarize conversation
async function summarizeConversation(userMessages, aiMessages) {
  if ((!userMessages || !userMessages.length) && (!aiMessages || !aiMessages.length)) {
    return null; // no messages to summarize
  }

  const text = [
    ...(userMessages || []).map(m => `User: ${m.message}`),
    ...(aiMessages || []).map(m => `AI: ${m.message}`)
  ].join("\n");

  return summarizeText(text);
}

export async function buildContext(id) {
  const allMessages = await fetchMessages(id) || [];

  // grab last 6 messages regardless of sender (or empty if chat just started)
  const lastSix = allMessages.slice(-6);
  const userMessages = lastSix.filter(m => m.sender === "user") || [];
  const aiMessages = lastSix.filter(m => m.sender === "ai") || [];

  // check if messages need summary
  const needsSummary = lastSix.some(m => !m.is_summarized);

  let summary = null;
  if (needsSummary && (userMessages.length || aiMessages.length)) {
    summary = await summarizeConversation(userMessages, aiMessages);

    if (summary) {
      await upsertSummary(id, summary);
      for (const msg of lastSix) {
        await setSummarized(msg.id);
      }
    }
  }

  const firstQuestion = allMessages.find(m => m.sender === "user")?.message || null;

  return {
    firstQuestion,
    userMessages,
    aiMessages,
    summary,
  };
}

// Batch summarizer
export async function summarizeBatch(chunks) {
  if (!Array.isArray(chunks)) {
    throw new Error('Expected "chunks" to be an array');
  }

  const summaries = await Promise.all(
    chunks.map(async (item) => {
      const summary = await summarizeText(item.raw_text || "");
      return {
        item_id: item.item_id,
        title: item.title,
        section: item.section,
        url: item.url,
        score: item.score ?? null,
        summary,
      };
    })
  );

  return summaries;
}
