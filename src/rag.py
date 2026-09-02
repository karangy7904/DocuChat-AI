from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .pdf_processor import Chunk


@dataclass
class Answer:
    text: str
    sources: list[Chunk]
    scores: list[float]
    latency_seconds: float


class RAGPipeline:
    def __init__(self, store, generator, top_k: int = 3):
        self.store = store
        self.generator = generator
        self.top_k = top_k

    def ask(self, question: str) -> Answer:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        started = perf_counter()
        results = self.store.search(question, self.top_k)
        chunks = [chunk for chunk, _ in results]
        return Answer(
            text=self.generator.generate(question, chunks),
            sources=chunks,
            scores=[score for _, score in results],
            latency_seconds=perf_counter() - started,
        )

