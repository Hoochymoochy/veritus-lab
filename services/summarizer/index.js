const express = require('express');
const axios = require('axios');
const app = express();
const port = process.env.PORT || 11600;

app.use(express.json());

const summarizeText = async (text) => {
  const prompt = `Summarize the following legal section for clarity:\n\n"${text}"`;

  const response = await axios.post('http://ollama:11434/api/generate', {
    model: 'mistral', // or any model you got loaded in Ollama
    prompt: prompt,
    stream: false
  });

  return response.data.response;
};

app.post('/summarize-batch', async (req, res) => {
  const { chunks } = req.body;

  if (!Array.isArray(chunks)) {
    return res.status(400).json({ error: 'Expected "items" to be an array' });
  }

  try {
    const summaries = await Promise.all(
      chunks.map(async (item) => {
        const summary = await summarizeText(item.raw_text || '');
        return {
          item_id: item.item_id,
          title: item.title,
          section: item.section,
          url: item.url,
          score: item.score ?? null,
          summary: summary
        };
      })
    );

    res.json({ results: summaries });
  } catch (err) {
    console.error('🔥 Summarizer error:', err.message);
    res.status(500).json({ error: 'Failed to summarize all items' });
  }
});

app.listen(port, () => {
  console.log(`🧠 Summarizer cookin’ on port ${port}`);
});
