from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
import traceback
from utils.scraper import define_scrapes
from urls import document
from utils.embeder import embed
from utils.pinecode import insert_pinecone_index, search_legal_docs

MAX_WORKERS = 5  # tune based on your CPU & network speed


def process_doc(config):
    """Handle scraping, embedding, and Pinecone insert for one doc."""
    name = config.get("name", "document")
    try:
        print(f"⚙️ Starting: {name}")

        # Step 1: scrape
        print(f"  └─ Step 1: Scraping...")
        result = define_scrapes(
            url=config["url"],
            name=name,
            jurisdiction=config.get("jurisdiction", "Federal"),
            layout=config.get("layout", "static")
        )

        if not result:
            print(f"❌ Skipped (no data): {name}")
            return None
        
        print(f"  └─ Step 2: Embedding...")
        # Step 2: embed
        result = embed(result)
        if not result:
            print(f"❌ Failed embedding: {name}")
            return None
        
        print(f"  └─ Step 3: Inserting to Pinecone...")
        # Step 3: Pinecone insert
        insert_pinecone_index(result)
        print(f"✅ Done: {name}")

        return result

    except Exception as e:
        print(f"💥 Error on {name}:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {e}")
        print(f"   Full traceback:")
        traceback.print_exc()
        return None


# def scrape_batch(urls_config):
#     total = len(urls_config)
#     print(f"\n🚀 Launching {total} parallel scrapes with {MAX_WORKERS} workers...\n")

#     results = []
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         future_to_config = {executor.submit(process_doc, cfg): cfg for cfg in urls_config}

#         for i, future in enumerate(as_completed(future_to_config), 1):
#             cfg = future_to_config[future]
#             name = cfg.get("name", "document")
#             try:
#                 result = future.result()
#                 if result:
#                     results.append(result)
#             except Exception as e:
#                 print(f"💀 Thread error on {name}:")
#                 print(f"   Error type: {type(e).__name__}")
#                 print(f"   Error message: {e}")
#                 traceback.print_exc()

#             print(f"[{i}/{total}] Finished: {name}")

#     print(f"\n🏁 All done! {len(results)}/{total} succeeded.\n")
#     return results


def scrape_batch(urls_config, limit=None):
    if limit:
        urls_config = urls_config[:limit]

    total = len(urls_config)
    print(f"\n🚀 Starting test batch of {total} docs...\n")

    results = []
    for i, cfg in enumerate(urls_config, 1):
        url = cfg["url"]
        name = cfg.get("name", "document")

        print(f"⚙️ [{i}/{total}] Scraping {name} ({url})")

        data = define_scrapes(
            url=url,
            name=name,
            jurisdiction=cfg.get("jurisdiction", "Federal"),
            layout=cfg.get("layout", "static")
        )

        insert_pinecone_index(embed(data))     
        print(f"✅ Finished: {name}")

    print(f"\n✅ Finished batch of {total}.\n")
    return results



if __name__ == "__main__":
    # results = scrape_batch(document[:1])
    print(search_legal_docs("law"))