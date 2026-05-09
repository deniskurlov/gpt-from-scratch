import pytest

from src.data import load_corpus, Tokenizer

@pytest.fixture(scope="module")
def text():
    return load_corpus()

@pytest.fixture(scope="module")
def tok(text):
    return Tokenizer(text)

def test_encoding_roundtrip(text, tok):
    assert tok.decode(tok.encode(text)) == text

def test_vocab_size(tok, text):
    assert tok.vocab_size == len(set(text))

def test_vocab(tok, text):
    assert tok.vocab == sorted(set(text))
