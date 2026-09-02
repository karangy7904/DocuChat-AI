from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import ExtractiveGenerator
from src.pdf_processor import Chunk, clean_text, split_text
from src.rag import RAGPipeline
from src.vector_store import KeywordStore


def main() -> None:
    assert clean_text("Docu-\nment   text") == "Document text"
    assert len(split_text("Useful sentence. " * 100, 2, 120, 20)) > 1
    chunk = Chunk("The warranty lasts twelve months.", 3, "p3-c1")
    pipeline = RAGPipeline(KeywordStore([chunk]), ExtractiveGenerator(), 1)
    answer = pipeline.ask("How many months does the warranty last?")
    assert "twelve months" in answer.text
    assert "[Page 3]" in answer.text
    summary = pipeline.ask("Give me a very short summary")
    assert "twelve months" in summary.text
    assert "[Page 3]" in summary.text
    print("DocuChat dependency-light checks passed")


if __name__ == "__main__":
    main()
