"""Book-level splitting: split on whole books rather than a ratio of tokens."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data import (
    build_split_dataloaders,
    book_token_counts,
    extract_epub_books,
    extract_epub_text,
    prepare_dataset,
    split_token_ids_by_books,
    tokenize,
)


def test_extract_epub_books_groups_chapters_by_book_opening(two_book_epub: Path) -> None:
    books = extract_epub_books(two_book_epub)

    assert len(books) == 2
    assert "Harry looked at Ron in book one" in books[0]
    assert "Hermione read the letter" in books[0]
    assert "Dobby warned Harry in book two" in books[1]
    assert "Dobby" not in books[0]


def test_book_token_counts_sum_to_the_whole_corpus(two_book_epub: Path) -> None:
    books = extract_epub_books(two_book_epub)

    counts = book_token_counts(books)

    assert sum(counts) == len(tokenize(extract_epub_text(two_book_epub)))


def test_split_token_ids_by_books_takes_whole_books_contiguously() -> None:
    ids = list(range(100))
    counts = [10, 20, 30, 40]

    splits = split_token_ids_by_books(ids, counts, train_books=2)

    assert splits["train"] == list(range(30))
    assert splits["val"] == list(range(30, 60))
    assert splits["test"] == list(range(60, 100))
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(ids)


def test_split_token_ids_by_books_rejects_a_count_mismatch() -> None:
    with pytest.raises(ValueError, match="token counts"):
        split_token_ids_by_books(list(range(50)), [10, 20], train_books=1)


def test_split_token_ids_by_books_rejects_impossible_book_counts() -> None:
    with pytest.raises(ValueError, match="at least"):
        split_token_ids_by_books(list(range(30)), [10, 20], train_books=5)


def test_book_split_keeps_books_out_of_each_others_splits(two_book_epub: Path) -> None:
    books = extract_epub_books(two_book_epub)
    counts = book_token_counts(books)
    ids = list(range(sum(counts)))

    splits = split_token_ids_by_books(ids, counts, train_books=1)

    assert len(splits["train"]) == counts[0]
    assert len(splits["val"]) == counts[1]
    assert splits["test"] == []


def test_build_split_dataloaders_honours_explicit_splits() -> None:
    splits = {"train": list(range(0, 60)), "val": list(range(100, 140)), "test": list(range(200, 240))}

    loaders = build_split_dataloaders(splits, sequence_length=4, batch_size=8)

    train_seen = torch.cat([inputs.flatten() for inputs, _ in loaders["train"]])
    val_seen = torch.cat([inputs.flatten() for inputs, _ in loaders["val"]])
    assert int(train_seen.max()) < 60
    assert int(val_seen.min()) >= 100
    assert int(val_seen.max()) < 140


def test_prepare_dataset_book_mode_records_the_splits(two_book_epub: Path, tmp_path: Path) -> None:
    result = prepare_dataset(
        two_book_epub,
        tmp_path,
        vocab_size=None,
        sequence_length=8,
        split_by="book",
        train_books=1,
    )

    payload = torch.load(tmp_path / "token_ids.pt", weights_only=False)
    assert payload["split_mode"] == "book"
    assert set(payload["splits"]) == {"train", "val", "test"}
    assert len(payload["splits"]["train"]) == result.book_token_counts[0]
    assert payload["splits"]["test"] == []


def test_prepare_dataset_ratio_mode_remains_the_default(fixture_epub: Path, tmp_path: Path) -> None:
    prepare_dataset(fixture_epub, tmp_path, vocab_size=None, sequence_length=4)

    payload = torch.load(tmp_path / "token_ids.pt", weights_only=False)
    assert payload["split_mode"] == "ratio"
    assert set(payload["splits"]) == {"train", "val", "test"}
