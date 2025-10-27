from sentence_transformers import SentenceTransformer
import re

# Load the cross-lingual embedder
model = SentenceTransformer("intfloat/multilingual-e5-large")


def chunk_legal_text(text, doc_title="Unknown Document", max_chunks=None):
    """
    Splits a flat legal text blob into article-based chunks.
    Only splits on actual article headers, not references.
    
    Args:
        text: The legal text to chunk
        doc_title: Document title for metadata
        max_chunks: Maximum number of chunks to return (None = all chunks)
    """
    chunks = []
    
    # More specific pattern that looks for article headers
    patterns = [
        r'(?:^|\n)(\s*Art\.\s*\d+[º°]?\s*[-–—.:])',  # Art. 123 - or Art. 123.
        r'(?:^|\n)(\s*Artigo\s+\d+[º°]?\s*[-–—.:])',  # Artigo 123 -
        r'(?:^|\n)(\s*ARTIGO\s+\d+[º°]?\s*[-–—.:])',  # ARTIGO 123 -
    ]
    
    articles = None
    matched_pattern = None
    
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
        if matches:
            matched_pattern = pattern
            print(f"✅ Matched pattern: {pattern}")
            print(f"   Found {len(matches)} article headers")
            
            # Limit matches if max_chunks specified
            if max_chunks:
                matches = matches[:max_chunks]
                print(f"   Limited to first {max_chunks} articles for testing\n")
            else:
                print()
            
            # Split text at article boundaries
            articles = []
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i+1].start() if i+1 < len(matches) else len(text)
                articles.append(text[start:end])
            break
    
    if articles is None:
        print("⚠️ No article pattern matched - using entire text as one chunk")
        articles = [text]
    
    for art in articles:
        art_clean = " ".join(art.split())
        
        # Extract article number
        art_number = "N/A"
        for num_pattern in [r'Art\.\s*(\d+)', r'Artigo\s+(\d+)', r'ARTIGO\s+(\d+)']:
            match = re.search(num_pattern, art_clean, flags=re.IGNORECASE)
            if match:
                art_number = match.group(1)
                break
        
        chunk_text = f"""Document: {doc_title}
Section: Preâmbulo/Artigo
Article: {art_number}
Text: {art_clean}"""
        
        metadata = {
            "document": doc_title,
            "section": "Preâmbulo/Artigo",
            "article": art_number,
            "text": art_clean
        }
        chunks.append((chunk_text, metadata))
    
    return chunks


def embed(scraper_result, preview_limit=3, max_chunks=None):
    """
    Embed the scraped legal document with proper chunking.
    
    Args:
        scraper_result: Dict with 'title', 'text', etc.
        preview_limit: Number of chunks to preview (0 = no preview)
        max_chunks: Maximum chunks to process (None = all, useful for testing)
    """
    try:
        if not scraper_result:
            print("⚠️ Warning: Empty scraper result")
            return None
        
        doc_title = scraper_result.get('title', 'Unknown Document')
        text_content = scraper_result.get('text', '')
        
        chunks = chunk_legal_text(text_content, doc_title=doc_title, max_chunks=max_chunks)
        
        # Show preview if requested
        if preview_limit > 0:
            print(f"\n📋 Preview of first {min(preview_limit, len(chunks))} chunks:")
            for i, (chunk_text, metadata) in enumerate(chunks[:preview_limit]):
                print(f"\n--- Chunk {i+1} ---")
                print(f"Article: {metadata['article']}")
                print(f"Text preview: {metadata['text'][:200]}...")
            print()
        
        # Prepare lists for batch processing
        chunk_texts = []
        chunk_metadata = []
        
        # Single loop: enrich metadata and collect texts
        for i, (chunk_text, metadata) in enumerate(chunks):
            metadata['type'] = scraper_result.get('type', '')
            metadata['jurisdiction'] = scraper_result.get('jurisdiction', '')
            metadata['source'] = scraper_result.get('source', '')
            
            chunk_texts.append(chunk_text)
            chunk_metadata.append(metadata)
            
            print(f"Processing chunk {i+1}/{len(chunks)}: {metadata}\n ")
        
        # Embed all at once (much faster than one-by-one)
        print(f"\n🔄 Generating embeddings for {len(chunk_texts)} chunks...")
        embeddings = model.encode(chunk_texts, convert_to_tensor=True)
        print(f"✅ Embeddings complete! Shape: {embeddings.shape}")

        scraper_result['embeddings'] = embeddings
        scraper_result['embedded_texts'] = chunk_texts
        scraper_result['embedding_metadata'] = chunk_metadata
        scraper_result['embedding_count'] = len(chunks)

        return scraper_result

    except Exception as e:
        print(f"❌ Embedding error: {e}")
        import traceback
        traceback.print_exc()
        return None

