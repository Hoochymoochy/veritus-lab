import axios from "axios";
import { fetchMessages, upsertSummary, setSummarized } from "./chat.js";

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";

// Single section summarizer with streaming
export async function summarizeText(text, onToken) {
  if (!text || text.trim() === "") return null;

  const prompt = `
You are a professional legal summarizer. Summarize the following conversation or legal text into a concise, clear, narrative-style summary.
- Focus on clarity and key points.
- Avoid lists; make it flow naturally.
- Keep it readable for someone without prior context.
- Highlight decisions, questions, or important info.

Text:
"""${text}"""
  `;

  const response = await axios.post(
    `${OLLAMA_URL}/api/generate`,
    { model: "mistral", prompt, stream: true },
    { responseType: "stream" }
  );

  let buffer = "";
  for await (const chunk of response.data) {
    buffer += chunk.toString();
    const parts = buffer.split("\n").filter(Boolean);

    for (let i = 0; i < parts.length; i++) {
      const line = parts[i].replace(/^data:\s*/, "").trim();
      if (!line) continue;

      if (line === "[DONE]") {
        onToken("[DONE]");
        return;
      }

      try {
        const json = JSON.parse(line);
        if (json.response) onToken(json.response);
      } catch {
        buffer = line; // partial JSON
      }
    }
    buffer = "";
  }

  onToken("[DONE]");
}

// Summarize conversation
export async function summarizeConversation(userMessages, aiMessages, onToken) {
  if ((!userMessages || !userMessages.length) && (!aiMessages || !aiMessages.length)) return null;

  const text = [
    ...(userMessages || []).map(m => `User: ${m.message}`),
    ...(aiMessages || []).map(m => `AI: ${m.message}`)
  ].join("\n");

  return summarizeText(text, onToken);
}

// Build chat context with summary
export async function buildContext(id, onToken = () => {}) {
  const allMessages = await fetchMessages(id) || [];
  const lastSix = allMessages.slice(-6);
  const userMessages = lastSix.filter(m => m.sender === "user");
  const aiMessages = lastSix.filter(m => m.sender === "ai");

  const needsSummary = lastSix.some(m => !m.is_summarized);
  let summary = null;

  if (needsSummary && (userMessages.length || aiMessages.length)) {
    summary = await summarizeConversation(userMessages, aiMessages, onToken);

    if (summary) {
      await upsertSummary(id, summary);
      for (const msg of lastSix) await setSummarized(msg.id);
    }
  }

  const firstQuestion = allMessages.find(m => m.sender === "user")?.message || null;

  return { firstQuestion, userMessages, aiMessages, summary };
}
