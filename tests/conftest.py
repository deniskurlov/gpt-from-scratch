import pytest

from src.data import load_corpus, Tokenizer


@pytest.fixture(scope="module")
def text():
    return load_corpus()


@pytest.fixture(scope="module")
def tok(text):
    return Tokenizer(text)
