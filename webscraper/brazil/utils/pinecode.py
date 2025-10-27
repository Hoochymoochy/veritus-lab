import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer


# 🧩 Load env vars from .env file
load_dotenv()

model = SentenceTransformer("intfloat/multilingual-e5-large")


def init_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not api_key or not index_name:
        raise ValueError("Missing Pinecone API key or index name in .env")

    # 🚀 Init Pinecone client
    pc = Pinecone(api_key=api_key)

    # 🧱 Create index if not exists
    existing_indexes = [index["name"] for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1024,  # multilingual-e5-large uses 1024
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    # 🔗 Connect to index
    return pc.Index(index_name)

index = init_pinecone()


def insert_pinecone_index(scraper_result):
    """
    Insert embedded legal documents into Pinecone.
    
    Args:
        scraper_result: Dictionary from embed() with structure:
            {
                "title": "Document Title",
                "type": "constitution/law/etc",
                "jurisdiction": "federal/state",
                "source": "https://...",
                "embeddings": torch.Tensor (shape: [num_chunks, 1024]),
                "embedded_texts": ["chunk1", "chunk2", ...],
                "embedding_metadata": [
                    {
                        "document": "...",
                        "section": "...",
                        "article": "1",
                        "text": "...",
                        "type": "...",
                        "jurisdiction": "...",
                        "source": "..."
                    },
                    ...
                ],
                "embedding_count": 123
            }
    """
    try:
        if not scraper_result or 'embeddings' not in scraper_result:
            print("⚠️ No embeddings found in scraper_result")
            return False
        
        embeddings = scraper_result['embeddings']
        embedded_texts = scraper_result.get('embedded_texts', [])
        metadata_list = scraper_result.get('embedding_metadata', [])
        doc_title = scraper_result.get('title', 'Unknown Document')
        
        # Convert PyTorch tensor to list
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy().tolist()
        
        # Validate dimensions
        if len(embeddings) != len(embedded_texts) or len(embeddings) != len(metadata_list):
            print(f"⚠️ Dimension mismatch: {len(embeddings)} embeddings, {len(embedded_texts)} texts, {len(metadata_list)} metadata")
            return False
        
        print(f"\n📤 Uploading to Pinecone: {doc_title}")
        print(f"   📊 Preparing {len(embeddings)} vectors")
        
        # Build vectors in Pinecone format
        vectors = []
        for idx, (embedding, text, metadata) in enumerate(zip(embeddings, embedded_texts, metadata_list)):
            # Create unique ID using document name and article number
            doc_name_clean = metadata.get('document', doc_title).replace(' ', '_').replace('/', '_')
            article_num = metadata.get('article', 'NA')
            vector_id = f"{doc_name_clean}_art{article_num}_{idx}"
            
            # Prepare metadata for Pinecone (keep under 40KB limit)
            pinecone_metadata = {
                'document': str(metadata.get('document', ''))[:200],
                'article': str(metadata.get('article', 'N/A')),
                'section': str(metadata.get('section', ''))[:100],
                'type': str(metadata.get('type', ''))[:50],
                'jurisdiction': str(metadata.get('jurisdiction', ''))[:50],
                'source': str(metadata.get('source', ''))[:300],
                'text': str(metadata.get('text', ''))[:1000],  # Truncate for size
                'chunk_index': idx
            }
            
            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': pinecone_metadata
            })
        
        if not vectors:
            print("⚠️ No vectors to insert")
            return False
        
        # Upsert to Pinecone in batches of 100
        batch_size = 100
        total_batches = (len(vectors) - 1) // batch_size + 1
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            batch_num = i // batch_size + 1
            print(f"   ✅ Batch {batch_num}/{total_batches}: {len(batch)} vectors")
        
        print(f"   🎉 Successfully inserted {len(vectors)} vectors for {doc_title}\n")
        return True
        
    except Exception as e:
        print(f"❌ Pinecone insertion error: {e}")
        import traceback
        traceback.print_exc()
        return False


def search_legal_docs(query_text, top_k=5, filter_dict=None):
    """
    Search legal documents using semantic similarity.
    
    Args:
        query_text: Search query string
        top_k: Number of results to return
        filter_dict: Optional Pinecone filter, e.g., {"type": "constitution"}
    
    Returns:
        List of matching results with metadata
    """
    
    try:
        # Embed the query using the same model
        query_embedding = model.encode([query_text], convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy().tolist()[0]
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Format and display results
        print(f"\n🔍 Search results for: '{query_text}'\n")
        matches = []
        
        for i, match in enumerate(results.get('matches', []), 1):
            metadata = match.get('metadata', {})
            score = match.get('score', 0)
            
            print(f"{i}. Article {metadata.get('article', 'N/A')} [Score: {score:.3f}]")
            print(f"   Document: {metadata.get('document', 'Unknown')}")
            print(f"   Type: {metadata.get('type', 'N/A')} | Jurisdiction: {metadata.get('jurisdiction', 'N/A')}")
            print(f"   Text: {metadata.get('text', '')[:200]}...")
            print(f"   Source: {metadata.get('source', 'N/A')}\n")
            
            matches.append({
                'id': match.get('id'),
                'score': score,
                'metadata': metadata
            })
        
        return matches
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        import traceback
        traceback.print_exc()
        return []