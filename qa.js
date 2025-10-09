// 🧠 Summarizer Module — Veritus style
import axios from "axios";

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";

// helper: format item into a clean citation
function formatCitation(item) {
  // try to extract section id from item_id ("title-73_chap-18_sec-15.5")
  const sectionMatch = item.item_id?.match(/title-(\d+)_chap-(\d+)_sec-(.+)/);
  let cite = "";
  if (sectionMatch) {
    cite = `Utah Code §${sectionMatch[1]}-${sectionMatch[2]}-${sectionMatch[3]}`;
  } else {
    cite = item.section || "Unknown Section";
  }

  return `According to ${cite} ([source](${item.url})): ${item.raw_text}`;
}

export async function finalResponse(items, question, context) {
  if (!Array.isArray(items)) {
    throw new Error('Expected "items" to be an array');
  }

  // Format retrieved items into smooth citations
  const flatSummaries = items.map(formatCitation).join("\n\n");

  const convoBits = [
    `First Question: ${context.firstQuestion || "N/A"}`,
    `Recent User: ${context.userMessages.map(m => m.message).join(" | ")}`,
    `Recent AI: ${context.aiMessages.map(m => m.message).join(" | ")}`,
    context.summary ? `Batch Summary: ${context.summary}` : "",
  ].join("\n");

  const groundingRule = `
    IMPORTANT: You must quote or paraphrase ONLY from the retrieved sections below.
    If a question cannot be answered directly using these retrieved laws, reply:
    "The retrieved sections do not cover this directly."
    Do NOT infer, expand, or reference any other Utah Code section or case law not explicitly included.
    `;


  const prompt = `
  You are VERITUS — a precise, citation-backed AI legal analyst.
  ${groundingRule}

  🎯 Rules: 
  - Respond strictly using the retrieved laws. 
  - If something's missing, say: "The retrieved sections do not cover this." 
  - Keep it tight (1–3 sentences). - Cite like: "According to [Jurisdiction Code §title-chapter-section]([source](url)), …" 
  - Never fabricate codes.

  ❓ User Question:
  "${question}"

  📚 Retrieved Sections:
  ${flatSummaries}

  🧩 Context (for style/tone awareness only):
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
