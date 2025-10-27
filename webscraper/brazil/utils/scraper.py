import os
import re
import unicodedata
from bs4 import BeautifulSoup
from utils.fetch import fetch_html_static, fetch_html_dynamic, fetch_pdf, get_links, get_links_tables
from utils.save import save_json
from utils.structure import parse_hierarchy

# Track visited URLs to prevent infinite loops
_visited_urls = set()

def define_scrapes(url, name, jurisdiction="Federal", layout="static"):
    """
    Main scraping function that fetches content and structures it hierarchically.
    
    Args:
        url: Target URL to scrape
        name: Document type/name for organization
        jurisdiction: Legal jurisdiction (default: "Federal")
        layout: Fetching strategy - "static", "dynamic", or "table"
    
    Returns:
        dict: Structured data with hierarchy, or None if failed
    """
    # Prevent infinite loops
    if url in _visited_urls:
        print(f"⏭️  Skipping already visited: {url}")
        return None
    _visited_urls.add(url)
    
    print(f"📜 Scraping {name} ({layout}) from {url}...")
    
    structured_data = None
    text_content = ""
    title_text = name.title()
    
    # Handle PDF documents
    if url.lower().endswith(".pdf"):
        text_content = fetch_pdf(url)
        if not text_content:
            print(f"❌ Failed to fetch PDF: {name}")
            return None
        # PDFs don't have HTML hierarchy
        structured_data = None
    
    # Handle HTML documents
    else:
        html = None
        
        # Choose fetching strategy
        if layout == "static":
            html = fetch_html_static(url)
        
        elif layout == "table":
            html = fetch_html_static(url)
            if html:
                # Extract and recursively scrape table links
                links = get_links_tables(html)
                print(f"🔗 Found {len(links)} links in tables")
                for link in links:
                    define_scrapes(link, name, jurisdiction, "static")
        
        elif layout == "dynamic":
            html = fetch_html_dynamic(url)
            if html:
                # Extract and recursively scrape all links
                links = get_links(html)
                print(f"🔗 Found {len(links)} links")
                for link in links:
                    define_scrapes(link, name, jurisdiction, "static")
        
        else:
            print(f"❌ Unsupported layout: {layout}")
            return None
        
        # Validate HTML was fetched
        if not html:
            print(f"❌ Failed to fetch HTML: {name}")
            return None
        
        # Parse HTML content
        soup = BeautifulSoup(html, "html.parser")
        
        # Extract title
        title_tag = soup.find("h1") or soup.find("title")
        title_text = title_tag.get_text(strip=True) if title_tag else name.title()
        
        # Extract plain text
        text_content = soup.get_text("\n", strip=True)
        
        # Parse hierarchical structure
        structured_data = parse_hierarchy(html)
        print(f"📊 Parsed {len(structured_data)} top-level sections")
    
        # Build metadata structure
        data = {
            "type": name,
            "title": make_ascii_id(title_text),
            "date": "Unknown",  # Could extract from metadata if available
            "jurisdiction": jurisdiction,
            "source": url,
            "text": text_content,
        }
    
    
    # Save to file
    output_dir = os.path.join("output", name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize filename
    safe_title = re.sub(r'[^\w\s-]', '', title_text)[:50]
    filename = f"{safe_title}_{hash(url) % 10000}.json"
    save_path = os.path.join(output_dir, filename)
    
    save_json(data, save_path, pretty=True)
    print(f"✅ Saved {data['title']} → {save_path}\n")
    
    return data


def make_ascii_id(text):
    # Remove accents
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    # Keep only letters, numbers, underscore, dash
    text = re.sub(r'[^a-zA-Z0-9_-]', '_', text)
    return text
