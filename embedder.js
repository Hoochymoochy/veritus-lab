// 🔥 Embedding + Pinecone Module (no HTTP)
// Clean + Modular — Veritus style.

//TODO have the namespcae link with country_city so it can be filtered on the root index we will call law. for now using "" will look throught all data in index

import axios from "axios";
import { Pinecone } from "@pinecone-database/pinecone";
import { v4 as uuidv4 } from "uuid";
import dotenv from "dotenv";

dotenv.config();

// 🌍 Env Checks
const { PINECONE_API_KEY, PINECONE_INDEX, OLLAMA_URL } = process.env;

if (!PINECONE_API_KEY || !PINECONE_INDEX) {
  throw new Error("❌ Missing required Pinecone env vars.");
}

// 🔐 Pinecone Init
const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
const index = pinecone.index(PINECONE_INDEX);

// ⚡ Helpers
const getOllamaUrl = () => OLLAMA_URL || "http://localhost:11434";

async function embedText(text) {
  const { data } = await axios.post(`${getOllamaUrl()}/api/embeddings`, {
    model: "nomic-embed-text",
    prompt: text,
  });
  return data.embedding;
}

// 🔍 Search Pinecone
export async function embedAndSearch(query, namespace = "") {
  if (!query) throw new Error("Query is required");

  const queryVector = await embedText(query);

  const queryRequest = {
    vector: queryVector,
    includeMetadata: true,
    includeValues: false,
    topK: 5,
    ...(namespace && { namespace }),
  };

  const result = await index.query(queryRequest);
  return result.matches.map(m => ({ score: m.score, ...m.metadata }));
}

// 📦 Embed + Store in Pinecone
export async function embedAndStore(input, metadata = {}, namespace = "") {
  if (!input) throw new Error("Input text is required");

  const vector = await embedText(input);
  const id = uuidv4();

  await index.upsert(
    [
      {
        id,
        values: vector,
        metadata: { text: input, ...metadata },
      },
    ],
    namespace ? { namespace } : {}
  );

  return { id, success: true };
}

// 📂 Bulk Upload JSON
export async function bulkUploadJson(json, namespace = "") {
  if (!Array.isArray(json)) throw new Error("JSON must be an array");

  const results = [];

  for (let i = 0; i < json.length; i++) {
    const item = json[i];
    const input =
      item.input ||
      item.text ||
      item.content ||
      item.metadata?.text ||
      item.body ||
      item.description ||
      item.section;

    if (!input) {
      results.push({ success: false, error: "No text content found", item: i });
      continue;
    }

    try {
      const vector = await embedText(input);
      const id = uuidv4();

      await index.upsert(
        [
          {
            id,
            values: vector,
            metadata: { ...item.metadata, raw_text: input, source: "json_upload" },
          },
        ],
        namespace ? { namespace } : {}
      );

      results.push({ success: true, id, text: input.slice(0, 100) });
    } catch (err) {
      results.push({ success: false, error: err.message, text: input.slice(0, 100) });
    }
  }

  const successCount = results.filter(r => r.success).length;
  return {
    message: `Uploaded ${successCount}/${json.length} items successfully`,
    results: results.slice(0, 10), // preview only
    summary: { total: json.length, successful: successCount, failed: json.length - successCount },
  };
}

// 💚 Health
export async function healthCheck() {
  return { status: "ok", timestamp: new Date().toISOString() };
}

// 🔗 Ollama Test
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
