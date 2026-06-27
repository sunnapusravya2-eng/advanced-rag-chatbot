import os
from io import BytesIO
from importlib import import_module
from typing import BinaryIO
from dotenv import load_dotenv
from pypdf import PdfReader

# Load environment variables
load_dotenv()

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    try:
        RecursiveCharacterTextSplitter = import_module(
            "langchain.text_splitter"
        ).RecursiveCharacterTextSplitter
    except ModuleNotFoundError:
        RecursiveCharacterTextSplitter = None

def extract_text_from_pdf(pdf_file: BinaryIO) -> str:
    """Extract all text from a PDF file-like object."""
    try:
        raw_bytes = pdf_file.read()
        if not raw_bytes:
            return ""

        reader = PdfReader(BytesIO(raw_bytes))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text.strip())

        return "\n\n".join(pages)
    except Exception as error:
        raise RuntimeError(f"Unable to read PDF file: {error}") from error


def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks using recursive splitting."""
    if not text:
        return []

    if RecursiveCharacterTextSplitter is None:
        chunks = []
        start = 0
        step = max(chunk_size - chunk_overlap, 1)

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start += step

        return chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)
