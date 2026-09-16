"""Corpus extraction, cleaning, tokenization, vocabulary, and sequence datasets.

Pipeline: EPUB -> clean text -> words/punctuation tokens -> integer IDs ->
contiguous train/validation/test split -> shifted fixed-length sequences.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# --- tokenizer -----------------------------------------------------------------

#: A word (optionally containing internal apostrophes) or a single punctuation mark.
TOKEN_PATTERN = re.compile(r"\w+(?:['\u2019]\w+)*|[^\w\s]")

#: Typographic characters mapped to their plain ASCII equivalents.
CHARACTER_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2032": "'", "\u2033": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": " - ", "\u2014": " - ",
    "\u2015": " - ", "\u2212": "-",
    "\u2026": "...",
    "\u00a0": " ", "\u2002": " ", "\u2003": " ", "\u2009": " ", "\u200a": " ",
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
    "\u00ad": "",
}

#: A line that is only a number (page number) or only punctuation (separator art).
ARTIFACT_LINE = re.compile(r"^[\W_]+$|^\d+$", re.UNICODE)

#: Project Gutenberg style boilerplate boundaries.
GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE)
GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE)

#: A word split by a line break, e.g. "exam-\nined" -> "examined".
LINEBREAK_HYPHEN = re.compile(r"(?<=[A-Za-z])-\n(?=[a-z])")


def tokenize(text: str) -> list[str]:
    """Split text into word and punctuation tokens, preserving case."""
    normalized = text
    for source, target in CHARACTER_MAP.items():
        normalized = normalized.replace(source, target)
    return TOKEN_PATTERN.findall(normalized)


# --- cleaning ------------------------------------------------------------------


def strip_boilerplate(text: str) -> str:
    """Keep only the text between Project Gutenberg START/END markers, if present."""
    start = GUTENBERG_START.search(text)
    if start:
        text = text[start.end():]
    end = GUTENBERG_END.search(text)
    if end:
        text = text[: end.start()]
    return text


def clean_text(raw: str) -> str:
    """Apply only the basic cleaning a word-level corpus needs.

    Keeps punctuation, capitalization, names, and terminology. Removes ebook
    metadata, extraction artifacts, and inconsistent whitespace.
    """
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = strip_boilerplate(text)
    text = LINEBREAK_HYPHEN.sub("", text)
    for source, target in CHARACTER_MAP.items():
        text = text.replace(source, target)

    kept: list[str] = []
    for line in text.split("\n"):
        collapsed = re.sub(r"[ \t\f\v]+", " ", line).strip()
        if collapsed and ARTIFACT_LINE.match(collapsed):
            continue
        kept.append(collapsed)

    joined = "\n".join(kept)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


#: Opening-chapter titles of the seven books, in series order. The omnibus EPUB
#: stores chapters, not books, so these are what mark a new book. Titles are
#: factual metadata; they carry no book text.
BOOK_OPENING_CHAPTERS = (
    "THE BOY WHO LIVED",
    "THE WORST BIRTHDAY",
    "OWL POST",
    "THE RIDDLE HOUSE",
    "DUDLEY DEMENTED",
    "THE OTHER MINISTER",
    "THE DARK LORD ASCENDING",
)


def extract_epub_chapters(epub_path: Path) -> list[tuple[str, str]]:
    """Return ``(heading, text)`` for every content document, in spine order."""
    import warnings

    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    from ebooklib import ITEM_DOCUMENT, epub

    book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})
    chapters: list[tuple[str, str]] = []
    for item_id, _linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ITEM_DOCUMENT:
            continue
        html = item.get_content().decode("utf-8", errors="replace")
        # Content documents are XHTML, but real EPUBs carry HTML-ish markup and
        # sometimes malformed XML. The HTML parser is the forgiving choice, so
        # suppress bs4's "XML parsed as HTML" notice instead of letting it fire
        # once per chapter.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav"]):
            tag.decompose()
        heading = ""
        for name in ("h1", "h2", "h3"):
            found = soup.find(name)
            if found:
                heading = found.get_text(" ", strip=True)
                break
        text = soup.get_text(separator="\n")
        if text.strip():
            chapters.append((heading, text.strip()))
    return chapters


def book_start_indices(chapters: Sequence[tuple[str, str]]) -> list[int]:
    """Indices of chapters that open a book, in reading order."""
    openings = {title.upper() for title in BOOK_OPENING_CHAPTERS}
    return [index for index, (heading, _) in enumerate(chapters) if heading.upper() in openings]


def extract_epub_books(epub_path: Path) -> list[str]:
    """Return the cleaned text of each book in the omnibus, in reading order.

    Front matter before the first recognised opening chapter (title page, table
    of contents, copyright) is dropped, since it is not prose. When no opening
    chapter is recognised the whole document is returned as a single book, which
    keeps single-book EPUBs working.
    """
    chapters = extract_epub_chapters(epub_path)
    if not chapters:
        return []

    starts = book_start_indices(chapters)
    if not starts:
        return [clean_text("\n\n".join(text for _heading, text in chapters))]

    ends = starts[1:] + [len(chapters)]
    return [
        clean_text("\n\n".join(text for _heading, text in chapters[start:end]))
        for start, end in zip(starts, ends)
    ]


def extract_epub_text(epub_path: Path) -> str:
    """Extract the readable text of an EPUB in spine (reading) order.

    This is the concatenation of :func:`extract_epub_books`, so the corpus and
    the book-level split are always derived from exactly the same text.
    """
    return "\n\n".join(extract_epub_books(epub_path))


# --- statistics ----------------------------------------------------------------


@dataclass(frozen=True)
class CorpusStats:
    """Word-level statistics of a cleaned corpus."""

    characters: int
    words: int
    unique_words: int
    most_common: list[tuple[str, int]] = field(default_factory=list)


def corpus_statistics(text: str, top_n: int = 20) -> CorpusStats:
    """Count characters, tokens, distinct tokens, and the most common tokens."""
    tokens = tokenize(text)
    counts = Counter(tokens)
    return CorpusStats(
        characters=len(text),
        words=len(tokens),
        unique_words=len(counts),
        most_common=counts.most_common(top_n),
    )


# --- vocabulary ----------------------------------------------------------------


@dataclass
class Vocabulary:
    """A word-level vocabulary with reserved ``<PAD>`` and ``<UNK>`` entries."""

    token_to_id: dict[str, int]
    id_to_token: list[str]

    pad_id: int = 0
    unk_id: int = 1

    @classmethod
    def build(cls, tokens: Iterable[str], max_size: int | None = None) -> Vocabulary:
        """Build a vocabulary ordered by frequency, then by first appearance."""
        counts: Counter[str] = Counter()
        first_seen: dict[str, int] = {}
        for index, token in enumerate(tokens):
            counts[token] += 1
            first_seen.setdefault(token, index)

        ranked = sorted(counts, key=lambda token: (-counts[token], first_seen[token]))
        id_to_token = [PAD_TOKEN, UNK_TOKEN]
        for token in ranked:
            if max_size is not None and len(id_to_token) >= max_size:
                break
            id_to_token.append(token)
        return cls(
            token_to_id={token: i for i, token in enumerate(id_to_token)},
            id_to_token=id_to_token,
        )

    def __len__(self) -> int:
        return len(self.id_to_token)

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Map tokens to IDs, sending anything unseen to ``<UNK>``."""
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        """Map IDs back to tokens."""
        return [self.id_to_token[int(i)] for i in ids]

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "hp-rnn-vocabulary-v1",
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
            "id_to_token": self.id_to_token,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        id_to_token = list(payload["id_to_token"])
        return cls(token_to_id={token: i for i, token in enumerate(id_to_token)},
                   id_to_token=id_to_token)


# --- dataset -------------------------------------------------------------------


def split_token_ids(
    ids: Sequence[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[list[int], list[int], list[int]]:
    """Split token IDs into contiguous train/validation/test sections."""
    total = len(ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return list(ids[:train_end]), list(ids[train_end:val_end]), list(ids[val_end:])


def book_token_counts(books: Sequence[str]) -> list[int]:
    """Token count of each book, in reading order."""
    return [len(tokenize(book)) for book in books]


def split_token_ids_by_books(
    ids: Sequence[int],
    counts: Sequence[int],
    train_books: int = 5,
) -> dict[str, list[int]]:
    """Split token IDs by whole books: first ``train_books`` train, then val, then test.

    Boundaries land between books rather than mid-sentence, so test is a genuinely
    held-out book instead of the tail of a document the model has partly seen.
    """
    if len(ids) != sum(counts):
        raise ValueError(
            f"token counts must sum to the number of token IDs "
            f"({sum(counts):,} vs {len(ids):,})"
        )
    if len(counts) < train_books + 1:
        raise ValueError(
            f"book splitting requires at least {train_books + 1} books for a train "
            f"and validation split, got {len(counts)}"
        )

    train_end = sum(counts[:train_books])
    val_end = train_end + counts[train_books]
    return {
        "train": list(ids[:train_end]),
        "val": list(ids[train_end:val_end]),
        "test": list(ids[val_end:]),
    }


def make_sequence_tensors(
    ids: Sequence[int],
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build shifted ``(input, target)`` tensors of shape ``[N, sequence_length]``."""
    if len(ids) <= sequence_length:
        raise ValueError("need more token IDs than sequence_length to build windows")
    data = torch.tensor(list(ids), dtype=torch.long)
    windows = data.unfold(0, sequence_length + 1, 1)
    return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()


class LanguageModelDataset(Dataset):
    """Sliding windows where the target is the input shifted by one token."""

    def __init__(self, ids: Sequence[int], sequence_length: int) -> None:
        self.inputs, self.targets = make_sequence_tensors(ids, sequence_length)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def build_split_dataloaders(
    splits: Mapping[str, Sequence[int]],
    sequence_length: int,
    batch_size: int,
) -> dict[str, DataLoader]:
    """Create one loader per named split, skipping splits too short to window."""
    loaders: dict[str, DataLoader] = {}
    for name, split_ids in splits.items():
        if len(split_ids) <= sequence_length:
            continue
        loaders[name] = DataLoader(
            LanguageModelDataset(split_ids, sequence_length),
            batch_size=batch_size,
            shuffle=(name == "train"),
            drop_last=False,
        )
    return loaders


def build_dataloaders(
    ids: Sequence[int],
    sequence_length: int,
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, DataLoader]:
    """Create train/validation/test loaders over one contiguous ratio split."""
    train_ids, val_ids, test_ids = split_token_ids(ids, train_ratio, val_ratio)
    return build_split_dataloaders(
        {"train": train_ids, "val": val_ids, "test": test_ids},
        sequence_length,
        batch_size,
    )


# --- preparation ---------------------------------------------------------------


def prepare_corpus(epub_path: Path, output_dir: Path) -> Path:
    """Extract and clean an EPUB into ``output_dir/corpus.txt``.

    Extraction already cleans each book, so the corpus is exactly the join of the
    books that the book-level split operates on.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.txt"
    corpus_path.write_text(extract_epub_text(Path(epub_path)) + "\n", encoding="utf-8")
    return corpus_path


def prepare_dataset(
    epub_path: Path,
    output_dir: Path,
    vocab_size: int | None,
    sequence_length: int,
    split_by: str = "ratio",
    train_books: int = 5,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> PreparedDataset:
    """Full Stage 2-4 preparation: corpus, vocabulary, token IDs, and splits."""
    if split_by not in ("ratio", "book"):
        raise ValueError(f"split_by must be 'ratio' or 'book', got {split_by!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    books = extract_epub_books(Path(epub_path))
    corpus_text = "\n\n".join(books)
    corpus_path = output_dir / "corpus.txt"
    corpus_path.write_text(corpus_text + "\n", encoding="utf-8")

    counts = book_token_counts(books)
    tokens = tokenize(corpus_text)
    vocab = Vocabulary.build(tokens, max_size=vocab_size)
    vocab.save(output_dir / "vocab.json")
    ids = vocab.encode(tokens)

    if split_by == "book":
        splits = split_token_ids_by_books(ids, counts, train_books)
    else:
        train_ids, val_ids, test_ids = split_token_ids(ids, train_ratio, val_ratio)
        splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    torch.save(
        {
            "token_ids": torch.tensor(ids, dtype=torch.long),
            "splits": splits,
            "split_mode": split_by,
            "book_token_counts": counts,
            "sequence_length": sequence_length,
            "vocab_size": vocab.size,
            "total_tokens": len(tokens),
        },
        output_dir / "token_ids.pt",
    )
    return PreparedDataset(
        corpus_path=corpus_path,
        tokens=tokens,
        vocab=vocab,
        ids=ids,
        splits=splits,
        book_token_counts=counts,
        stats=corpus_statistics(corpus_text),
    )


@dataclass(frozen=True)
class PreparedDataset:
    """Everything :func:`prepare_dataset` produced, ready to train on."""

    corpus_path: Path
    tokens: list[str]
    vocab: Vocabulary
    ids: list[int]
    splits: dict[str, list[int]]
    book_token_counts: list[int]
    stats: CorpusStats


def format_statistics(stats: CorpusStats, vocab_size: int, sequence_length: int) -> str:
    """Render corpus statistics the way the project brief asks for them."""
    lines = [
        f"total characters : {stats.characters:,}",
        f"total words      : {stats.words:,}",
        f"unique words     : {stats.unique_words:,}",
        f"vocabulary size  : {vocab_size:,} (including <PAD> and <UNK>)",
        f"context length   : {sequence_length}",
        "most common words:",
    ]
    lines += [f"  {token:<14} {count:,}" for token, count in stats.most_common]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.data", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="build corpus.txt, vocab.json, and token_ids.pt")
    prepare.add_argument("--input", type=Path, required=True, help="local EPUB file")
    prepare.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    prepare.add_argument("--vocab-size", type=int, default=18000)
    prepare.add_argument("--sequence-length", type=int, default=64)
    prepare.add_argument(
        "--split-by",
        choices=("book", "ratio"),
        default="book",
        help="'book' holds out whole books (default); 'ratio' cuts the token stream 80/10/10",
    )
    prepare.add_argument(
        "--train-books",
        type=int,
        default=5,
        help="with --split-by book: books 1..N train, book N+1 validation, the rest test",
    )

    inspect = sub.add_parser("inspect", help="print one shifted batch from the loaders")
    inspect.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    inspect.add_argument("--batch-size", type=int, default=4)
    inspect.add_argument("--sequence-length", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "prepare":
        result = prepare_dataset(
            args.input,
            args.output_dir,
            args.vocab_size,
            args.sequence_length,
            split_by=args.split_by,
            train_books=args.train_books,
        )
        splits = result.splits
        counts = result.book_token_counts
        print(format_statistics(result.stats, result.vocab.size, args.sequence_length))
        print(f"\nbooks detected   : {len(counts)}")
        for index, count in enumerate(counts, start=1):
            print(f"  book {index}: {count:,} tokens")
        print(f"\nsplit mode       : {args.split_by}")
        for name in ("train", "val", "test"):
            size = len(splits[name])
            print(f"  {name:<6} {size:>10,} tokens  ({size / max(1, len(result.ids)):.1%})")
        print(f"\nwrote {result.corpus_path}")
        print(f"wrote {args.output_dir / 'vocab.json'}")
        print(f"wrote {args.output_dir / 'token_ids.pt'}")
        return 0

    if args.command == "inspect":
        payload = torch.load(args.processed_dir / "token_ids.pt", weights_only=False)
        vocab = Vocabulary.load(args.processed_dir / "vocab.json")
        sequence_length = args.sequence_length or int(payload["sequence_length"])
        loaders = build_dataloaders(
            payload["token_ids"].tolist(), sequence_length, args.batch_size
        )
        inputs, targets = next(iter(loaders["train"]))
        print(f"input  shape {tuple(inputs.shape)}")
        print(f"target shape {tuple(targets.shape)}\n")
        for row in range(min(args.batch_size, inputs.shape[0])):
            print("input :", " ".join(vocab.decode(inputs[row].tolist())))
            print("target:", " ".join(vocab.decode(targets[row].tolist())))
            aligned = bool(torch.equal(inputs[row, 1:], targets[row, :-1]))
            print(f"shifted-by-one: {aligned}\n")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
