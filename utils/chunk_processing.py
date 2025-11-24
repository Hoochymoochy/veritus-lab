"""
Utilities for processing and formatting document chunks.
"""


def ensure_chunk_metadata(chunks):
    """
    Ensure all chunks have properly extracted URLs in metadata.
    
    Args:
        chunks: List of chunk dictionaries
    
    Returns:
        List of processed chunks with proper metadata
    """
    processed_chunks = []
    for chunk in chunks:
        # Make a copy to avoid mutating original
        processed = dict(chunk)
        
        if 'metadata' not in processed:
            processed['metadata'] = {}
        
        metadata = processed['metadata']
        
        # Try multiple ways to extract URL
        url = (
            metadata.get('url') or 
            metadata.get('source_url') or 
            metadata.get('source') or 
            chunk.get('source') or 
            chunk.get('url') or
            ''
        )
        
        # Ensure URL is a string and looks like a planalto URL
        if url and isinstance(url, str) and 'planalto.gov.br' in url.lower():
            metadata['url'] = url
        else:
            metadata['url'] = ''
            
        processed_chunks.append(processed)
    
    return processed_chunks


def format_context_chunk(chunk, index):
    """
    Format a single context chunk with metadata and reference number.
    
    Args:
        chunk: Chunk dictionary with text and metadata
        index: Reference index number
    
    Returns:
        Formatted string representation of the chunk
    """
    metadata = chunk.get('metadata', {})
    text = chunk.get('text') or chunk.get('raw_text') or ''
    
    source = metadata.get('source', 'Unknown')
    doc_type = metadata.get('type', 'N/A')
    country = metadata.get('country', 'N/A')
    state = metadata.get('state', 'N/A')
    
    # CRITICAL: Extract URL from metadata or construct from source
    url = metadata.get('url') or metadata.get('source_url') or ''
    
    # If no URL in metadata but we have a source field that looks like a URL
    if not url and source and 'planalto.gov.br' in str(source).lower():
        url = source
    
    # Create structured context with clear reference number
    formatted = f"""
[REFERENCE {index + 1}]
📘 Source Document: {source}
🔗 EXACT URL TO CITE: {url if url else '[URL NOT AVAILABLE IN DATABASE]'}
🧾 Type: {doc_type} | Jurisdiction: {country}/{state}

Content:
{text}

[END REFERENCE {index + 1}]
"""
    return formatted.strip()