import express from 'express'
import axios from 'axios'

const app = express()
app.use(express.json())

app.post('/ask', async (req, res) => {
  const { query } = req.body

  try {
    // 1. 🔍 Embed + Retrieve
    const { data: embedderResult } = await axios.post('http://embedder:11500/search', { query })

    const chunks = embedderResult.results || embedderResult.chunks || embedderResult

    if (!Array.isArray(chunks)) {
      return res.status(400).json({ error: '❌ Embedder did not return an array of chunks' })
    }

    console.log('📦 Retrieved Chunks:', chunks.length)

    // 2. 🧠 Summarize Retrieved Chunks
    const { data: summaryResult } = await axios.post('http://summarizer:11600/summarize-batch', {
      chunks
    })

    console.log('📝 Summary Complete')

    // 3. 🤖 Ask LLM (optional – uncomment when ready)
    const { data: answerResult } = await axios.post('http://qa:11700/final-response', {
      chunks,
      question: query
    })

    // 4. ✅ Verify Answer (optional)
    // const { data: verifiedResult } = await axios.post('http://verifier:5053/verify', {
    //   answer: answerResult
    // })

    // 5. 🌐 Simplify/Translate (optional)
    // const { data: finalResult } = await axios.post('http://explainer:5054/explain', {
    //   text: verifiedResult
    // })

    res.json({
      summary: answerResult,
      // answer: answerResult,
      // verified: verifiedResult,
      // final: finalResult
    })
  } catch (err) {
    console.error('🔥 Orchestrator Error:', err.message)
    res.status(500).json({ error: 'Something broke in the orchestration flow.' })
  }
})

app.listen(4000, () => {
  console.log('🧠 Veritus-Lab orchestrator cookin’ on port 4000')
})
