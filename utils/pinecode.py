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


def search_legal_docs(
    query_text,
    top_k=8,
    context=None,
    country=None,
    state=None,
    filter_dict=None
):
    """
    🔎 Search legal documents with contextual precision and query enhancement.

    Args:
        query_text (str): The legal or natural language query.
        top_k (int): Max number of matches to return.
        context (str): Chat history/context to enrich the query.
        country (str): Country code or name (for jurisdiction filtering).
        state (str): State or region (for finer jurisdiction).
        filter_dict (dict): Optional Pinecone filter overrides.

    Returns:
        list[dict]: Ranked and formatted search matches.
    """
    try:
        # --- Build dynamic filters ---
        filters = {}

        if country:
            filters["country"] = {"$eq": country.lower()}
        if state:
            filters["state"] = {"$eq": state.lower()}

        # Merge any manual filter overrides
        if filter_dict:
            filters.update(filter_dict)

        # --- Enhanced query with context ---
        enhanced_query = query_text
        
        if context:
            # Extract key legal terms and entities from context
            context_snippet = context[-500:] if len(context) > 500 else context
            
            # Create a context-aware query (prioritize current query)
            enhanced_query = f"{query_text}\n\nRelated context: {context_snippet}"
            
            print(f"🧠 Context-enhanced query created (length: {len(enhanced_query)})")

        # --- Embed the enhanced query ---
        query_embedding = model.encode([enhanced_query], convert_to_tensor=True)
        query_vector = query_embedding.cpu().numpy().tolist()[0]

        # --- Query Pinecone with higher top_k for reranking ---
        search_k = top_k * 2 if context else top_k
        
        results = index.query(
            vector=query_vector,
            top_k=search_k,
            include_metadata=True,
            filter=filters or None,
        )

        matches = []
        print(f"\n🧭 Contextual Search for: '{query_text}' [{country}/{state}]\n")

        for i, match in enumerate(results.get("matches", []), 1):
            metadata = match.get("metadata", {})
            score = float(match.get("score", 0))
            jurisdiction = (
                f"{metadata.get('country', '')}/{metadata.get('state', '')}".strip("/")
            )
            snippet = metadata.get("text", "")[:220].replace("\n", " ")

            if i <= top_k or not context:  # Only print top_k results
                print(f"{i}. {metadata.get('title', 'Unknown')} [Score: {score:.3f}]")
                print(f"   Jurisdiction: {jurisdiction or 'N/A'}")
                print(f"   Type: {metadata.get('type', 'N/A')}")
                print(f"   Snippet: {snippet}...")
                print(f"   Source: {metadata.get('source', 'N/A')}\n")

            matches.append({
                "id": match.get("id"),
                "score": score,
                "metadata": metadata,
                "text": metadata.get("text", ""),
                "title": metadata.get("title", "Unknown"),
                "chapter": metadata.get("chapter"),
                "section": metadata.get("section"),
            })

        # --- Context-aware reranking ---
        if context and len(matches) > top_k:
            print("🔄 Applying context-aware reranking...")
            
            from difflib import SequenceMatcher
            
            # Extract keywords from original query and context
            query_keywords = set(query_text.lower().split())
            context_keywords = set(context.lower().split()) if context else set()
            
            def context_boost(m):
                base_score = m["score"]
                text = m.get("text", "").lower()
                
                # Keyword overlap with query (high weight)
                query_overlap = len(query_keywords & set(text.split())) / max(len(query_keywords), 1)
                
                # Keyword overlap with context (medium weight)
                context_overlap = len(context_keywords & set(text.split())) / max(len(context_keywords), 1)
                
                # Text similarity with query
                text_sim = SequenceMatcher(None, query_text.lower(), text[:500]).ratio()
                
                # Combined boost
                boost_factor = (
                    1 + 
                    (query_overlap * 0.3) + 
                    (context_overlap * 0.15) + 
                    (text_sim * 0.1)
                )
                
                return base_score * boost_factor
            
            matches.sort(key=context_boost, reverse=True)
            matches = matches[:top_k]
            
            print(f"✅ Reranked to top {top_k} context-relevant results\n")

        return matches

    except Exception as e:
        print(f"❌ Search error: {e}")
        import traceback
        traceback.print_exc()
        return []


# --- Alternative: Two-stage retrieval for better context usage ---

def search_legal_docs_two_stage(
    query_text,
    top_k=8,
    context=None,
    country=None,
    state=None,
    filter_dict=None
):
    """
    🔎 Two-stage search: first pass with context, second pass with original query.
    Combines results for better context-aware retrieval.
    """
    try:
        filters = {}
        if country:
            filters["country"] = {"$eq": country.lower()}
        if state:
            filters["state"] = {"$eq": state.lower()}
        if filter_dict:
            filters.update(filter_dict)

        all_matches = {}  # Use dict to deduplicate by ID
        
        # --- Stage 1: Context-enhanced search ---
        if context:
            context_snippet = context[-400:] if len(context) > 400 else context
            enhanced_query = f"{query_text}. Previous discussion: {context_snippet}"
            
            print("🔍 Stage 1: Context-enhanced search...")
            context_embedding = model.encode([enhanced_query], convert_to_tensor=True)
            context_vector = context_embedding.cpu().numpy().tolist()[0]
            
            context_results = index.query(
                vector=context_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filters or None,
            )
            
            for match in context_results.get("matches", []):
                match_id = match.get("id")
                all_matches[match_id] = {
                    "id": match_id,
                    "score": float(match.get("score", 0)) * 1.1,  # Boost context matches
                    "metadata": match.get("metadata", {}),
                    "text": match.get("metadata", {}).get("text", ""),
                    "title": match.get("metadata", {}).get("title", "Unknown"),
                    "chapter": match.get("metadata", {}).get("chapter"),
                    "section": match.get("metadata", {}).get("section"),
                    "source": "context"
                }
        
        # --- Stage 2: Original query search ---
        print("🔍 Stage 2: Direct query search...")
        query_embedding = model.encode([query_text], convert_to_tensor=True)
        query_vector = query_embedding.cpu().numpy().tolist()[0]
        
        query_results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filters or None,
        )
        
        for match in query_results.get("matches", []):
            match_id = match.get("id")
            if match_id in all_matches:
                # Average the scores if found in both stages
                all_matches[match_id]["score"] = (
                    all_matches[match_id]["score"] + float(match.get("score", 0))
                ) / 2
                all_matches[match_id]["source"] = "both"
            else:
                all_matches[match_id] = {
                    "id": match_id,
                    "score": float(match.get("score", 0)),
                    "metadata": match.get("metadata", {}),
                    "text": match.get("metadata", {}).get("text", ""),
                    "title": match.get("metadata", {}).get("title", "Unknown"),
                    "chapter": match.get("metadata", {}).get("chapter"),
                    "section": match.get("metadata", {}).get("section"),
                    "source": "query"
                }
        
        # Sort by score and return top_k
        matches = sorted(all_matches.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        
        print(f"\n🧭 Combined Search Results: '{query_text}' [{country}/{state}]\n")
        for i, match in enumerate(matches, 1):
            print(f"{i}. {match.get('title')} [Score: {match['score']:.3f}] (Source: {match.get('source')})")
            print(f"   Snippet: {match.get('text', '')[:200]}...\n")
        
        return matches

    except Exception as e:
        print(f"❌ Search error: {e}")
        import traceback
        traceback.print_exc()
        return []