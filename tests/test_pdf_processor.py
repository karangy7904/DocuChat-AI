import pytest

from src.pdf_processor import clean_text, split_text


def test_clean_text_repairs_line_break_hyphenation():
    assert clean_text("Docu-\nment   text") == "Document text"


def test_split_text_preserves_page():
    chunks = split_text("A useful sentence. " * 100, page=4, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    assert all(chunk.page == 4 for chunk in chunks)


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        split_text("text", page=1, chunk_size=100, overlap=100)

