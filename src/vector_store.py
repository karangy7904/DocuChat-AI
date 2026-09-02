from __future__ import annotations

from typing import Protocol

import numpy as np

from .pdf_processor import Chunk


class Encoder(Protocol):
    def encode(self, texts: list[str], normalize_embeddings: bool = True): ...


class MiniLMEncoder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], normalize_embeddings: bool = True):
        return self.model.encode(
            texts, normalize_embeddings=normalize_embeddings, show_progress_bar=False
        )


class FaissVectorStore:
    def __init__(self, encoder: Encoder):
        self.encoder = encoder
        self.index = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        import faiss

        if not chunks:
            raise ValueError("At least one text chunk is required")
        vectors = np.asarray(
            self.encoder.encode([item.text for item in chunks], normalize_embeddings=True),
            dtype="float32",
        )
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = chunks

    def search(self, question: str, k: int = 3) -> list[tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("Upload and process a PDF first")
        query = np.asarray(
            self.encoder.encode([question], normalize_embeddings=True), dtype="float32"
        )
        scores, indexes = self.index.search(query, min(k, len(self.chunks)))
        return [
            (self.chunks[int(index)], float(score))
            for index, score in zip(indexes[0], scores[0])
            if index >= 0
        ]


class KeywordStore:
    """Tiny offline fallback used by unit tests."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def search(self, question: str, k: int = 3) -> list[tuple[Chunk, float]]:
        terms = {word.lower().strip(".,?!") for word in question.split() if len(word) > 2}
        ranked = []
        for chunk in self.chunks:
            words = {word.lower().strip(".,?!") for word in chunk.text.split()}
            ranked.append((chunk, len(terms & words) / max(1, len(terms))))
        return sorted(ranked, key=lambda item: item[1], reverse=True)[:k]

