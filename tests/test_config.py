import pytest

from localrag.config import load_config


def test_defaults_when_file_missing(tmp_path):
    cfg = load_config(str(tmp_path / "nope.yaml"))
    assert cfg.chunk_size == 800


def test_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("chunk_size: 500\nchunk_overlap: 50\ntop_k: 3\n")
    cfg = load_config(str(p))
    assert (cfg.chunk_size, cfg.top_k) == (500, 3)


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("chunk_sizee: 500\n")
    with pytest.raises(ValueError):
        load_config(str(p))


def test_bad_overlap_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("chunk_size: 100\nchunk_overlap: 100\n")
    with pytest.raises(ValueError):
        load_config(str(p))
