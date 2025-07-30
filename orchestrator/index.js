// orchestrator/index.ts (Node/Express example)
import express from 'express'
import axios from 'axios'

const app = express()
app.use(express.json())

app.post('/ask', async (req, res) => {
  const { query } = req.body

  // 1. Embed + retrieve
  const chunks = await axios.post('http://embedder:11500/search', { query })

  // 2. Summarize chunks
  // const summary = await axios.post('http://localhost:11600/summarize-batch', { chunks })
  

  // 3. Main answer from LLM
  // const answer = await axios.post('http://localhost:5052/ask', { summary, question })

  // // 4. Verify answer
  // const verified = await axios.post('http://localhost:5053/verify', { answer })

  // // 5. Translate/simplify
  // const final = await axios.post('http://localhost:5054/explain', { text: verified })

  res.json({ chunks: chunks.data })
})

app.listen(4000, () => console.log('🧠 Veritus-Lab orchestrator on port 4000'))
