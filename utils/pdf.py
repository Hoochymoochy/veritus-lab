import requests
import pdfplumber

def fetch_pdf_text(url, retries=3, delay=3):
    """
    Fetch a PDF, extract all text.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            with open("temp.pdf", "wb") as f:
                f.write(response.content)
            
            text = ""
            with pdfplumber.open("temp.pdf") as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            os.remove("temp.pdf")
            return text
        except Exception as e:
            print(f"⚠️ PDF fetch error ({attempt+1}/{retries}): {e}")
            sleep(delay)
    return None
