import axios from "axios";
import { Pinecone } from "@pinecone-database/pinecone";
import { v4 as uuidv4 } from "uuid";
import dotenv from "dotenv";
import { finalResponse } from "./finalResponse.js"; // streaming AI function

dotenv.config();

const { PINECONE_API_KEY, PINECONE_INDEX, OLLAMA_URL } = process.env;
if (!PINECONE_API_KEY || !PINECONE_INDEX) throw new Error("❌ Missing Pinecone env vars.");

const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
const index = pinecone.index(PINECONE_INDEX);
const getOllamaUrl = () => OLLAMA_URL || "http://localhost:11434";

// ⚡ Core embedding
export async function embedText(text) {
  const { data } = await axios.post(`${getOllamaUrl()}/api/embeddings`, {
    model: "nomic-embed-text",
    prompt: text,
  });
  return data.embedding;
}

// 🔍 Pinecone search
export async function embedAndSearch(query, namespace = "", context = {}) {
  if (!query) throw new Error("Query required");

  let contextText = "";
  if (context) {
    const { summary, userMessages, aiMessages } = context;
    contextText = summary ? `Summary:\n${summary}\n` : "";
    if (userMessages?.length || aiMessages?.length) {
      contextText += [
        ...(userMessages || []).map(m => `User: ${m.message}`),
        ...(aiMessages || []).map(m => `AI: ${m.message}`)
      ].join("\n");
    }
  }

  const fullPrompt = contextText ? `Context:\n${contextText}\nUser Question: ${query}` : query;
  const queryVector = await embedText(fullPrompt);

  const result = await index.query({
    vector: queryVector,
    includeMetadata: true,
    includeValues: false,
    topK: 5,
    ...(namespace && { namespace }),
  });

  return result.matches.map(m => ({ score: m.score, ...m.metadata }));
}

// 📦 Store embeddings
export async function embedAndStore(input, metadata = {}, namespace = "") {
  const vector = await embedText(input);
  const id = uuidv4();

  await index.upsert([{ id, values: vector, metadata: { text: input, ...metadata } }], 
                     namespace ? { namespace } : {});
  return { id, success: true };
}

// 🏎 Incremental embedding + streaming
export async function incrementalEmbedAndStream(texts, query, chatContext, onToken) {
  const contextChunks = [];

  for (let i = 0; i < texts.length; i++) {
    const text = texts[i];
    const vector = await embedText(text);
    contextChunks.push({ text, vector });

    if (contextChunks.length === 1) {
      finalResponse(contextChunks, query, chatContext, onToken).catch(err => {
        console.error("🔥 Stream error:", err);
      });
    }
  }

  if (contextChunks.length > 1) {
    await finalResponse(contextChunks, query, chatContext, onToken);
  }
}

// 💚 Health check
export async function healthCheck() {
  return { status: "ok", timestamp: new Date().toISOString() };
}

// 🔗 Ollama test
export async function testOllama() {
  try {
    const { data } = await axios.post(`${getOllamaUrl()}/api/embeddings`, {
      model: "nomic-embed-text",
      prompt: "test connection",
    });
    return { status: "ok", vectorLength: data.embedding?.length || 0 };
  } catch (err) {
    return { status: "fail", details: err.message };
  }
}
