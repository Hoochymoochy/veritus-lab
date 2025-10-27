# webscraper/brazil/utils/fetch.py
import requests
from time import sleep
from playwright.sync_api import sync_playwright, Error as PlaywrightError
import subprocess
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

HEADLESS = True

def ensure_playwright_browsers_installed():
    """Make sure Playwright browsers are ready to roll."""
    try:
        subprocess.run(["playwright", "install", "chromium"], check=True)
    except Exception as e:
        print(f"⚠️ Browser install check failed: {e}")


def fetch_html_static(url, retries=3, delay=3, timeout=25):
    """Basic HTTP request fallback."""
    for i in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            return response.text
        except Exception as e:
            print(f"⚠️ Error fetching {url} (attempt {i+1}/{retries}): {e}")
            sleep(delay)
    return None


def fetch_html_dynamic(url, retries=3, delay=3):
    """Fetch dynamic HTML using headless browser with retries."""
    ensure_playwright_browsers_installed()

    for attempt in range(1, retries + 1):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=HEADLESS, args=["--no-sandbox"])
                context = browser.new_context(user_agent=HEADERS["User-Agent"])
                page = context.new_page()
                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(2000)

                html = page.content()
                browser.close()
                return html
        except PlaywrightError as e:
            print(f"⚠️ Playwright fetch error (attempt {attempt}/{retries}): {e}")
            sleep(delay)
        except Exception as e:
            print(f"❌ Unexpected error (attempt {attempt}/{retries}): {e}")
            sleep(delay)
    return None

def fetch_pdf(url, retries=3, delay=3):
    import pdfplumber, os
    for i in range(retries):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            temp_path = "temp.pdf"
            with open(temp_path, "wb") as f:
                f.write(r.content)
            text = ""
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            os.remove(temp_path)
            return text
        except Exception as e:
            print(f"⚠️ PDF fetch error ({i+1}/{retries}): {e}")
            sleep(delay)
    return None

def get_links(html):
    """Extract all <a> links inside #content-core."""
    soup = BeautifulSoup(html, "html.parser")
    content_core = soup.find(id="content-core")
    links = []
    if not content_core:
        return links

    for a in content_core.find_all("a", href=True):
        href = a['href'].strip()
        if href.startswith("http"):
            links.append(href)
    return links


def get_links_tables(html):
    """Extract <a> links inside <table> elements within #content-core."""
    soup = BeautifulSoup(html, "html.parser")
    content_core = soup.find(id="content-core")
    links = []
    if not content_core:
        return links

    for table in content_core.find_all("table"):
        for a in table.find_all("a", href=True):
            href = a['href'].strip()
            if href.startswith("http"):
                links.append(href)
    return links