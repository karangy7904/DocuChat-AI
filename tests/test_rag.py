import pytest

from src.generator import ExtractiveGenerator
from src.pdf_processor import Chunk
from src.rag import RAGPipeline
from src.vector_store import KeywordStore


def pipeline():
    chunk = Chunk("The warranty lasts twelve months.", 3, "p3-c1")
    return RAGPipeline(KeywordStore([chunk]), ExtractiveGenerator(), 1)


def test_answer_contains_page_citation():
    answer = pipeline().ask("How many months does the warranty last?")
    assert "twelve months" in answer.text
    assert "[Page 3]" in answer.text


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        pipeline().ask(" ")


def test_summary_request_returns_grounded_excerpt():
    answer = pipeline().ask("Give me a very short summary")
    assert "warranty lasts twelve months" in answer.text
    assert "[Page 3]" in answer.text
