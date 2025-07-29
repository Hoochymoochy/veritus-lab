const express = require('express');
const axios = require('axios');
const { Pinecone } = require('@pinecone-database/pinecone');
const { v4: uuidv4 } = require('uuid');
const fileUpload = require("express-fileupload");
const app = express();

// Load environment variables first
require('dotenv').config();

app.use(fileUpload());
app.use(express.json());

// Debug: Check if environment variables are loaded
console.log('Environment check:');
console.log('PINECONE_API_KEY:', process.env.PINECONE_API_KEY ? 'Found' : 'Missing');
console.log('PINECONE_INDEX:', process.env.PINECONE_INDEX ? 'Found' : 'Missing');

// Validate required environment variables
if (!process.env.PINECONE_API_KEY) {
  console.error('❌ PINECONE_API_KEY is not set in environment variables');
  process.exit(1);
}

if (!process.env.PINECONE_INDEX) {
  console.error('❌ PINECONE_INDEX is not set in environment variables');
  process.exit(1);
}

// 🔐 Pinecone init
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX);

async function embedText(text) {
  const ollamaUrl = process.env.OLLAMA_URL || 'http://host.docker.internal:11434';
  const response = await axios.post(`${ollamaUrl}/api/embeddings`, {
    model: 'nomic-embed-text',
    prompt: text
  });
  return response.data.embedding;
}

app.post('/search', async (req, res) => {
  try {
    const { query, topK = 5, namespace = "" } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    // Step 1: Embed the query
    const ollamaUrl = process.env.OLLAMA_URL || 'http://host.docker.internal:11434';
    const embedRes = await axios.post(`${ollamaUrl}/api/embeddings`, {
      model: 'nomic-embed-text',
      prompt: query
    });

    const queryVector = embedRes.data.embedding;

    // Step 2: Query Pinecone
    const queryRequest = {
      topK,
      vector: queryVector,
      includeMetadata: true,
      includeValues: false,
    };

    // Only add namespace if it's not empty
    if (namespace) {
      queryRequest.namespace = namespace;
    }

    const result = await index.query(queryRequest);

    const matches = result.matches.map(match => ({
      score: match.score,
      ...match.metadata,
    }));

    res.json(matches);
  } catch (err) {
    console.error('Search error:', err.message);
    res.status(500).json({ error: 'Search failed', details: err.message });
  }
});

app.post('/embed', async (req, res) => {
  try {
    const { input, metadata = {}, namespace = "" } = req.body;

    if (!input) {
      return res.status(400).json({ error: 'Input text is required' });
    }

    // Step 1: Embed
    const ollamaUrl = process.env.OLLAMA_URL || 'http://host.docker.internal:11434';
    const embedRes = await axios.post(`${ollamaUrl}/api/embeddings`, {
      model: 'nomic-embed-text',
      prompt: input
    });

    const vector = embedRes.data.embedding;
    const id = uuidv4();

    // Step 2: Upsert into Pinecone
    const upsertRequest = [
      {
        id,
        values: vector,
        metadata: {
          text: input,
          ...metadata,
        }
      }
    ];

    const upsertOptions = {};
    if (namespace) {
      upsertOptions.namespace = namespace;
    }

    await index.upsert(upsertRequest, upsertOptions);

    res.json({ id, success: true });

  } catch (err) {
    console.error('Embed error:', err.message);
    res.status(500).json({ error: 'Embedding or Pinecone insert failed', details: err.message });
  }
});

app.post("/upload-json", async (req, res) => {
  if (!req.files || !req.files.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  let json;
  try {
    json = JSON.parse(req.files.file.data.toString());
  } catch (err) {
    return res.status(400).json({ error: "Invalid JSON file" });
  }

  if (!Array.isArray(json)) {
    return res.status(400).json({ error: "JSON must be an array of objects" });
  }

  const results = [];
  // Fix: req.body might be undefined with file uploads, use optional chaining
  const namespace = req.body?.namespace || req.query?.namespace || "";

  console.log(`Processing ${json.length} items from JSON file`);
  
  // Debug: Show structure of first item
  if (json.length > 0) {
    console.log('First item structure:', JSON.stringify(json[0], null, 2));
    console.log('Available keys:', Object.keys(json[0]));
  }

  for (let i = 0; i < json.length; i++) {
    const item = json[i];
    
    // Extract text from the correct location based on your JSON structure
    const input = item.input || item.text || item.content || 
                  item.metadata?.text || item.metadata?.content ||
                  item.body || item.description || item.section;
    
    // Use the existing metadata, but don't include the text field twice
    const metadata = {
      ...item.metadata,
      item_id: item.id, // Preserve the original ID
    };
    
    // Remove text from metadata since we're storing it separately
    if (metadata.text) {
      delete metadata.text;
    }

    if (!input) {
      if (i < 3) { // Only log first 3 for debugging
        console.log(`Skipping item ${i}, available keys:`, Object.keys(item));
      }
      results.push({ success: false, error: "No text content found", item: i, availableKeys: Object.keys(item) });
      continue;
    }

    try {
      console.log(`Processing item ${i + 1}/${json.length}: ${input.substring(0, 50)}...`);
      
      const vector = await embedText(input);
      const id = uuidv4();
      
      const upsertRequest = [
        {
          id,
          values: vector,
          metadata: {
            ...metadata,
            raw_text: input,
            source: 'json_upload'
          }
        }
      ];

      const upsertOptions = {};
      if (namespace) {
        upsertOptions.namespace = namespace;
      }

      await index.upsert(upsertRequest, upsertOptions);
      
      results.push({ 
        success: true, 
        id: id,
        text: input.substring(0, 100) + (input.length > 100 ? '...' : '')
      });
    } catch (err) {
      console.error(`Error processing item ${i}:`, err.message);
      results.push({ 
        success: false, 
        error: err.message, 
        text: input.substring(0, 100) + (input.length > 100 ? '...' : '')
      });
    }
  }

  const successCount = results.filter(r => r.success).length;
  console.log(`Upload complete: ${successCount}/${json.length} items processed successfully`);

  res.json({ 
    message: `Processed JSON file: ${successCount}/${json.length} items uploaded successfully`, 
    results: results.slice(0, 10), // Only return first 10 results to avoid huge response
    summary: {
      total: json.length,
      successful: successCount,
      failed: json.length - successCount
    }
  });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Debug endpoint to test Ollama connection
app.get('/test-ollama', async (req, res) => {
  try {
    const ollamaUrl = process.env.OLLAMA_URL || 'http://host.docker.internal:11434';
    console.log('Testing Ollama at:', ollamaUrl);
    
    const response = await axios.post(`${ollamaUrl}/api/embeddings`, {
      model: 'nomic-embed-text',
      prompt: 'test connection'
    });
    
    res.json({ 
      status: 'Ollama connection successful', 
      url: ollamaUrl,
      vectorLength: response.data.embedding?.length || 0
    });
  } catch (err) {
    console.error('Ollama test failed:', err.message);
    res.status(500).json({ 
      error: 'Ollama connection failed', 
      url: process.env.OLLAMA_URL || 'http://host.docker.internal:11434',
      details: err.message 
    });
  }
});

const PORT = process.env.PORT || 11500;
app.listen(PORT, () => {
  console.log(`Custom embedding server running on port ${PORT}`);
});