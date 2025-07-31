const express = require('express');
const axios = require('axios');
const app = express();
const port = process.env.PORT || 11700;

app.use(express.json());

const summarizeText = async (items, question) => {
  const flatSummaries = items
    .map(item => `• ${item.section}: ${item.summary}`)
    .join('\n');

  const prompt = `You’re a sharp legal AI. Based on the data below, provide a clear, concise final summary answering this question:
  \n"${question}"\n\n${flatSummaries}`;

  const response = await axios.post('http://ollama:11434/api/generate', {
    model: 'phi3', // Use the actual name used in `ollama list`
    prompt: prompt,
    stream: false
  });

  return response.data.response;
};

app.post('/final-response', async (req, res) => {
  const { chunks, question } = req.body;

  if (!Array.isArray(chunks)) {
    return res.status(400).json({ error: 'Expected "chunks" to be an array' });
  }

  try {
    const finalSummary = await summarizeText(chunks, question);
    res.json({ summary: finalSummary });
  } catch (err) {
    console.error('🔥 Summarizer error:', err.message);
    res.status(500).json({ error: 'Failed to summarize all chunks' });
  }
});

app.listen(port, () => {
  console.log(`🧠 Summarizer cookin’ on port ${port}`);
});
