import express from 'express'
import cors from 'cors'
import { embedAndSearch } from './embedder.js'
import { summarizeBatch, buildContext } from './summarizer.js'
import { finalResponse } from './qa.js'

const app = express()
app.use(express.json())
app.use(cors())

app.post('/ask', async (req, res) => {
  const { query, id } = req.body

  try {
    // 1. 🔍 Embed + Retrieve
    const chunks = await embedAndSearch(query)
    if (!Array.isArray(chunks)) {
      return res.status(400).json({ error: '❌ Embedder did not return an array of chunks' })
    }

    console.log('📦 Retrieved Chunks:', chunks.length)

    // 2. 🧠 Summarize
    const summary = await summarizeBatch(chunks)
    const chatContext = await buildContext(id)
    console.log('📝 Summary Complete')
    console.log(summary)

    // 3. 🤖 Ask LLM,
    const answer = await finalResponse(chunks, query, chatContext);

    res.json({
      summary,
      answer
    })
  } catch (err) {
    console.error('🔥 Orchestrator Error:', err.message)
    res.status(500).json({ error: 'Something broke in the orchestration flow.' })
  }
})

app.listen(4000, "0.0.0.0", () => {
  console.log('🧠 Veritus-Lab orchestrator cookin’ on port 4000')
})
