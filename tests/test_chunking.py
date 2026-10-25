import pytest

from localrag.chunking import chunk_text, normalize


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n ") == []


def test_short_text_is_single_chunk():
    assert chunk_text("hello world", size=100, overlap=10) == ["hello world"]


def test_chunks_respect_size_and_cover_text():
    text = " ".join(f"word{i}" for i in range(600))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 3
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[0].startswith("word0")
    assert chunks[-1].endswith("word599")


def test_overlap_is_shared_between_neighbours():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_text(text, size=150, overlap=50)
    tail_words = set(chunks[0].split()[-3:])
    assert tail_words & set(chunks[1].split())


def test_prefers_paragraph_boundaries():
    para = "A" * 120
    text = f"{para}\n\n{para}\n\n{para}"
    chunks = chunk_text(text, size=200, overlap=20)
    assert chunks[0] == para


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


def test_normalize_collapses_whitespace():
    assert normalize("a  \t b\r\n\n\n\nc") == "a b\n\nc"
