import re
from bs4 import BeautifulSoup

def parse_hierarchy(html):
    """
    Parse Brazilian legal text into structured hierarchy:
    Livro -> Título -> Capítulo -> Seção -> Artigo
    Handles messy or truncated pages gracefully.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(['s', 'strike', 'script', 'style']):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    data = []
    root_level = {"articles": []}
    current_livro = current_title = current_chapter = current_section = None
    last_article = None

    for line in lines:
        # LIVRO
        if re.match(r'^LIVRO\s+[IVXLC]+', line, re.IGNORECASE):
            if current_livro:
                data.append(current_livro)
            current_livro = {"livro": line, "titles": [], "chapters": [], "articles": []}
            current_title = current_chapter = current_section = None
            last_article = None

        # TÍTULO
        elif re.match(r'^T[IÍ]TULO\s+[IVXLC]+', line, re.IGNORECASE):
            current_title = {"title": line, "chapters": [], "sections": [], "articles": []}
            if current_livro:
                current_livro["titles"].append(current_title)
            else:
                data.append(current_title)
            current_chapter = current_section = None
            last_article = None

        # CAPÍTULO
        elif re.match(r'^CAP[IÍ]TULO\s+[IVXLC]+', line, re.IGNORECASE):
            current_chapter = {"chapter": line, "sections": [], "articles": []}
            if current_title:
                current_title["chapters"].append(current_chapter)
            elif current_livro:
                current_livro["chapters"].append(current_chapter)
            else:
                data.append(current_chapter)
            current_section = None
            last_article = None

        # SEÇÃO
        elif re.match(r'^(SUB)?SE[CÇ][ÃA]O\s+[IVXLC]+', line, re.IGNORECASE):
            current_section = {"section": line, "articles": []}
            if current_chapter:
                current_chapter["sections"].append(current_section)
            elif current_title:
                current_title.setdefault("sections", []).append(current_section)
            elif current_livro:
                current_livro.setdefault("sections", []).append(current_section)
            else:
                data.append(current_section)
            last_article = None

        # ARTIGO
        elif re.match(r'^Art\.?\s*\d+[º°]?', line, re.IGNORECASE):
            art_num = re.findall(r'^Art\.?\s*\d+[º°]?', line, re.IGNORECASE)[0]
            article = {"art_number": art_num, "text": line}
            last_article = article

            if current_section:
                current_section["articles"].append(article)
            elif current_chapter:
                current_chapter["articles"].append(article)
            elif current_title:
                current_title.setdefault("articles", []).append(article)
            elif current_livro:
                current_livro.setdefault("articles", []).append(article)
            else:
                root_level["articles"].append(article)

        # Continuations or broken lines
        else:
            if last_article:
                if not line.upper().startswith("ART"):
                    last_article["text"] = " ".join([last_article["text"], line])
            else:
                root_level.setdefault("preamble", []).append(line)

    # Final append
    if current_livro:
        data.append(current_livro)

    if root_level["articles"] or root_level.get("preamble"):
        data.append(root_level)

    return clean_empty_keys(data)


def clean_empty_keys(obj):
    """Remove empty lists/dicts from nested structure"""
    if isinstance(obj, dict):
        return {k: clean_empty_keys(v) for k, v in obj.items() 
                if v or isinstance(v, (int, float, bool))}
    elif isinstance(obj, list):
        cleaned = [clean_empty_keys(item) for item in obj]
        return [item for item in cleaned if item]
    return obj