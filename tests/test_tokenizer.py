import pytest

from core.tokenizer import BPETokenizer


def test_unicode_round_trip(tmp_path):
    text = "Transformers learn patterns. नमस्ते"
    tokenizer = BPETokenizer.train([text] * 4, target_vocab_size=280)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = BPETokenizer.load(path)
    assert restored.encode(text) == tokenizer.encode(text)


def test_unknown_token_is_rejected():
    with pytest.raises(ValueError, match="unknown token"):
        BPETokenizer().decode([999])
