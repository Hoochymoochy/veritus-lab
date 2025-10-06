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
    // 1️⃣ Grab previous context (chat history or summary)
    const chatContext = await buildContext(id)

    // 2️⃣ Search Pinecone with context-fused embedding
    const chunks = await embedAndSearch(query, "", chatContext)

    if (!Array.isArray(chunks)) {
      return res.status(400).json({ error: '❌ Embedder did not return an array of chunks' })
    }

    console.log('📦 Retrieved Chunks:', chunks.length)

    // 3️⃣ Summarize retrieved info
    const summary = await summarizeBatch(chunks)

    console.log('📝 Chunks Summary finish')
  

    // 4️⃣ Ask LLM with contextual blend
    const answer = await finalResponse(summary, query, chatContext)

    console.log('📝 Final Answer finish')

    res.json({ summary, answer })
  } catch (err) {
    console.error('🔥 Orchestrator Error:', err.message)
    res.status(500).json({ error: 'Something broke in the orchestration flow.' })
  }
})



app.listen(4000, "0.0.0.0", () => {
  console.log('🧠 Veritus-Lab orchestrator cookin’ on port 4000')
})
