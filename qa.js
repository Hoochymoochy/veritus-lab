// 🧠 Summarizer Module — Veritus style
import axios from "axios";

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";

export async function finalResponse(items, question, context) {
  if (!Array.isArray(items)) {
    throw new Error('Expected "items" to be an array');
  }

  const flatSummaries = items
    .map(item => `• ${item.section || "Unknown Section"}: ${item.summary || item.text || ""}`)
    .join("\n");

  const convoBits = [
    `First Question: ${context.firstQuestion || "N/A"}`,
    `Recent User: ${context.userMessages.map(m => m.message).join(" | ")}`,
    `Recent AI: ${context.aiMessages.map(m => m.message).join(" | ")}`,
    context.summary ? `Batch Summary: ${context.summary}` : "",
  ].join("\n");

  const prompt = `
You are a sharp legal AI. Based on the retrieved sections + chat context, answer clearly:

❓ Question: "${question}"

📚 Retrieved Sections:
${flatSummaries}

💬 Conversation Context:
${convoBits}
  `;

  const response = await axios.post(`${OLLAMA_URL}/api/generate`, {
    model: "phi3", // swap to your local model
    prompt,
    stream: false,
  });

  return {
    summary: response.data.response,
    urls: items.map(item => item.url).filter(Boolean),
  };
}
