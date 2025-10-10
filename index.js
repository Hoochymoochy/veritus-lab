import express from 'express';
import cors from 'cors';
import { embedAndSearch, incrementalEmbedAndStream } from './embedder.js';
import { buildContext } from './summarizer.js';

const app = express();
app.use(express.json());
app.use(cors());

app.post('/ask', async (req, res) => {
  const { query, id } = req.body;

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  try {
    const chatContext = await buildContext(id);

    const chunks = await embedAndSearch(query, "", chatContext);

    if (!Array.isArray(chunks)) {
      res.write(`data: ${JSON.stringify({ error: "Bad chunks" })}\n\n`);
      return res.end();
    }


    // 🏎 Stream AI response as embeddings are processed
    await incrementalEmbedAndStream(
      chunks.map(c => {
      const meta = `Source: ${c.chapter || c.title || "Unknown"}\nSection: ${c.section || "N/A"}\nURL: ${c.url || "N/A"}\n\n`
      return meta + (c.text || c.raw_text || "")}),
      query,
      chatContext,
      (token) => {
        res.write(`data: ${JSON.stringify({ token })}\n\n`);
        if (token === "[DONE]") res.end();
      }
    );

  } catch (err) {
    console.error('🔥 Orchestrator Error:', err.message);
    res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
    res.end();
  }
});

app.listen(4000, "0.0.0.0", () => {
  console.log('🧠 Veritus-Lab orchestrator running on port 4000');
});

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    timestamp: new Date().toISOString(),
    message: "Backend streaming ready ✅",
  });
});