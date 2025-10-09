import axios from "axios";

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";

/**
 * Streams AI response token by token
 * @param {Array} chunks - Array of text/context chunks
 * @param {string} query - User question
 * @param {object} chatContext - Optional summary + messages
 * @param {function} onToken - Callback for each token
 */
export async function finalResponse(chunks, query, chatContext, onToken) {
  const contextText = chunks.map(c => c.text || c.raw_text || "").join("\n");
  const summaryText = chatContext.summary ? `Summary:\n${chatContext.summary}\n` : "";

  const prompt = `
You are a professional legal AI assistant. Use the context below to answer the user's question clearly and concisely.

Context:
${summaryText}${contextText}

User Question: ${query}
Answer in a narrative style, highlighting important decisions or info. Stream token by token.
  `;

  const response = await axios.post(
    `${OLLAMA_URL}/api/generate`,
    { model: "mistral", prompt, stream: true },
    { responseType: "stream" }
  );

  let buffer = "";
  for await (const chunk of response.data) {
    buffer += chunk.toString();
    const lines = buffer.split("\n").filter(Boolean);

    for (let line of lines) {
      line = line.replace(/^data:\s*/, "").trim();
      if (!line) continue;

      if (line === "[DONE]") {
        onToken("[DONE]");
        return;
      }

      try {
        const json = JSON.parse(line);
        if (json.response) onToken(json.response);
      } catch {
        buffer = line; // keep partial JSON for next chunk
      }
    }
    buffer = "";
  }

  onToken("[DONE]");
}

export default finalResponse;