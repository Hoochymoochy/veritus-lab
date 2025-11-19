from fastapi import UploadFile
import pdfplumber
import docx
import logging
import io


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


async def extract_text_from_file_bytes(content: bytes, filename: str) -> str:
    """Extract text from file bytes"""
    logging.info(f"Processing file: {filename}")
    filename_lower = filename.lower()

    # PDF
    if filename_lower.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)

    # DOCX
    if filename_lower.endswith(".docx"):
        doc = docx.Document(io.BytesIO(content))
        return "\n".join([para.text for para in doc.paragraphs])

    # TXT
    if filename_lower.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")

    logging.warning(f"Unsupported file type: {filename}")
    return ""


def clean_text(text: str) -> str:
    text = text.replace("\r", "")
    text = " ".join(text.split())
    return text