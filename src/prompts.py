from __future__ import annotations

from .pdf_processor import Chunk


SYSTEM_PROMPT = """You are DocuChat AI, a document question-answering assistant.
Answer using only the supplied document context.
Do not invent facts, names, dates, or numbers.
If the answer is unavailable, reply exactly: "I could not find this information in the supplied document."
Keep the answer under 100 words and cite relevant pages as [Page N]."""


def build_messages(question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    context = "\n\n".join(f"Page {chunk.page}:\n{chunk.text}" for chunk in chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
    ]

