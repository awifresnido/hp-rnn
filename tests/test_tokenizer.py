"""Stage 3 tests: word-level tokenization, vocabulary, and ID round-trip."""

from __future__ import annotations

from src.data import Vocabulary, tokenize


def test_tokenize_separates_words_and_punctuation() -> None:
    assert tokenize("Harry looked at Ron.") == ["Harry", "looked", "at", "Ron", "."]


def test_tokenize_keeps_contractions_and_capitalization() -> None:
    assert tokenize("Harry didn't see Hermione's wand!") == [
        "Harry",
        "didn't",
        "see",
        "Hermione's",
        "wand",
        "!",
    ]


def test_tokenize_handles_multicharacter_punctuation() -> None:
    assert tokenize('"Stop!"... he said.') == [
        '"',
        "Stop",
        "!",
        '"',
        ".",
        ".",
        ".",
        "he",
        "said",
        ".",
    ]


def test_vocabulary_reserves_pad_and_unk_ids() -> None:
    vocab = Vocabulary.build(["Harry", "Ron", "Harry"], max_size=10)

    assert vocab.pad_id == 0
    assert vocab.unk_id == 1
    assert vocab.id_to_token[0] == "<PAD>"
    assert vocab.id_to_token[1] == "<UNK>"


def test_vocabulary_orders_by_frequency_then_first_appearance() -> None:
    vocab = Vocabulary.build(["b", "b", "a", "c", "a"])

    assert vocab.token_to_id["b"] == 2
    assert vocab.token_to_id["a"] == 3
    assert vocab.token_to_id["c"] == 4


def test_vocabulary_respects_max_size_and_maps_rare_words_to_unk() -> None:
    vocab = Vocabulary.build(["a", "a", "b", "b", "c", "d"], max_size=4)

    assert vocab.size == 4
    assert vocab.encode(["a", "b", "c", "d"]) == [2, 3, 1, 1]


def test_encode_decode_round_trip() -> None:
    vocab = Vocabulary.build(["Harry", "looked", "at", "Ron", "."])
    tokens = ["Harry", "looked", "at", "Ron", "."]

    assert vocab.decode(vocab.encode(tokens)) == tokens


def test_vocabulary_save_and_load_round_trip(tmp_path) -> None:
    vocab = Vocabulary.build(["Harry", "Ron", "Hermione"])
    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = Vocabulary.load(path)

    assert loaded.token_to_id == vocab.token_to_id
    assert loaded.encode(["Harry", "Hagrid"]) == vocab.encode(["Harry", "Hagrid"])
