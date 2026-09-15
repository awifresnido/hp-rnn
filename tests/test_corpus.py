"""Stage 2 tests: EPUB extraction, basic cleaning, and corpus statistics."""

from __future__ import annotations

from pathlib import Path

from src.data import clean_text, corpus_statistics, extract_epub_text, prepare_corpus


def test_extract_epub_text_follows_spine_order(fixture_epub: Path) -> None:
    text = extract_epub_text(fixture_epub)

    assert "Harry looked at Ron" in text
    assert "Hermione" in text
    # Spine order: chapter one precedes chapter two.
    assert text.index("Harry") < text.index("Hermione")


def test_extract_epub_text_drops_markup_and_navigation(fixture_epub: Path) -> None:
    text = extract_epub_text(fixture_epub)

    assert "<p>" not in text and "<h1>" not in text
    assert "Chapter One" in text
    assert "One</a>" not in text


def test_clean_text_normalizes_typographic_quotes() -> None:
    cleaned = clean_text("“Hello,” she said — softly.\u00a0‘Yes.’")

    assert '"Hello," she said - softly. \'Yes.\'' == cleaned


def test_clean_text_joins_line_break_hyphenation() -> None:
    assert clean_text("Hermione exam-\nined the map.") == "Hermione examined the map."


def test_clean_text_drops_page_numbers_and_artifact_lines() -> None:
    cleaned = clean_text("Real paragraph.\n\n12\n\n***\n\nAnother paragraph.")

    assert cleaned == "Real paragraph.\n\nAnother paragraph."


def test_clean_text_collapses_excess_whitespace() -> None:
    assert clean_text("a  b\tc\r\n\r\n\r\n\r\nd") == "a b c\n\nd"


def test_corpus_statistics_counts_characters_words_and_uniques() -> None:
    stats = corpus_statistics("Harry looked at Ron . Harry smiled .")

    assert stats.characters == 36
    assert stats.words == 8
    assert stats.unique_words == 6
    assert stats.most_common[0] == ("Harry", 2)


def test_prepare_corpus_writes_processed_files(fixture_epub: Path, tmp_path: Path) -> None:
    corpus_path = prepare_corpus(fixture_epub, tmp_path)

    assert corpus_path == tmp_path / "corpus.txt"
    written = corpus_path.read_text(encoding="utf-8")
    assert "Harry looked at Ron" in written
    assert "examined the map" in written
    assert "***" not in written
