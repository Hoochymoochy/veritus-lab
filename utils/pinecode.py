import os
from pinecone import Pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 🧩 Load env vars
load_dotenv()

model = SentenceTransformer("intfloat/multilingual-e5-large")


def init_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    pc = Pinecone(api_key=api_key)
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
    """
    try:
        # --- Build dynamic filters ---
        filters = {}
        or_filters = []
        if state:
            or_filters.append({"state": {"$eq": state}})
        or_filters.append({"state": {"$eq": "Federal"}})

        filters["$or"] = or_filters

        # Merge manual filters
        if filter_dict:
            filters.update(filter_dict)

        # --- Enhance query with context ---
        enhanced_query = query_text
        if context:
            snippet = context[-500:] if isinstance(context, str) else str(context)[:500]
            enhanced_query = f"{query_text}\n\nRelated context: {snippet}"

        # --- Embed the query ---
        query_embedding = model.encode([enhanced_query], convert_to_tensor=True)
        query_vector = query_embedding.cpu().numpy().tolist()[0]

        # --- Query Pinecone ---
        search_k = top_k * 2 if context else top_k
        results = index.query(
            vector=query_vector,
            top_k=search_k,
            include_metadata=True,
            filter=filters or None
        )

        matches = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            matches.append({
                "id": match.get("id"),
                "score": float(match.get("score", 0)),
                "metadata": metadata,
                "text": metadata.get("text", ""),
                "title": metadata.get("title", "Unknown"),
                "chapter": metadata.get("chapter"),
                "section": metadata.get("section"),
            })

        # --- Optional context-aware reranking ---
        if context and len(matches) > top_k:
            from difflib import SequenceMatcher

            query_keywords = set(query_text.lower().split())
            context_text = context if isinstance(context, str) else ' '.join(str(v) for v in context.values())
            context_keywords = set(context_text.lower().split())

            def context_boost(m):
                base_score = m["score"]
                text = m.get("text", "").lower()
                query_overlap = len(query_keywords & set(text.split())) / max(len(query_keywords), 1)
                context_overlap = len(context_keywords & set(text.split())) / max(len(context_keywords), 1)
                text_sim = SequenceMatcher(None, query_text.lower(), text[:500]).ratio()
                return base_score * (1 + query_overlap*0.3 + context_overlap*0.15 + text_sim*0.1)

            matches.sort(key=context_boost, reverse=True)
            matches = matches[:top_k]

        return matches

    except Exception as e:
        print(f"❌ Search error: {e}")
        import traceback
        traceback.print_exc()
        return []
