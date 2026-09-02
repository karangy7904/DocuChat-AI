from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    page: int
    chunk_id: str


def clean_text(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, page: int, chunk_size: int = 600, overlap: int = 100) -> list[Chunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    text = clean_text(text)
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        part = text[start:end].strip()
        if part:
            chunks.append(Chunk(part, page, f"p{page}-c{len(chunks) + 1}"))
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def extract_pdf(file_bytes: bytes) -> tuple[list[Chunk], int]:
    import fitz

    chunks: list[Chunk] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as document:
        page_count = len(document)
        for page_number, page in enumerate(document, start=1):
            chunks.extend(split_text(page.get_text("text"), page_number))
    if not chunks:
        raise ValueError("No selectable text found. Scanned PDFs require OCR, which this simple demo omits.")
    return chunks, page_count

