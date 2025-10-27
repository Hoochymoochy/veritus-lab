# webscraper/brazil/utils/clean.py
from bs4 import BeautifulSoup
import re

def clean_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator='\n')
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def get_art(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.find(id='art').text

def get_art(html):
    """
    Extracts the main legal content inside the div#art section of the page.
    Returns the cleaned text with preserved line breaks.
    """
    soup = BeautifulSoup(html, 'html.parser')
    art_div = soup.find(id='art')

    if not art_div:
        print("⚠️ No <div id='art'> found.")
        return ""

    # Extract all paragraphs and bolded headers (Title, Chapter, etc.)
    parts = []
    for tag in art_div.find_all(['p', 'b']):
        text = tag.get_text(" ", strip=True)
        if text:
            parts.append(text)

    return "\n".join(parts)
