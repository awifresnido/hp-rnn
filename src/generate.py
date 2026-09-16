"""Next-word prediction and recursive text generation.

Prediction and generation are different activities. Predicting reports a
probability distribution over the next word; generating feeds one choice back in
and repeats, so a small per-step error compounds into the final paragraph.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from src.data import Vocabulary, tokenize
from src.model import RecurrentLanguageModel

#: Tokens rendered without a leading space, because they attach to the prior word.
CLOSING_PUNCTUATION = frozenset({".", ",", "!", "?", ";", ":", ")", "]", "}", "..."})


@dataclass
class Prediction:
    """One candidate next word and its probability."""

    token: str
    probability: float


def tokenize_prompt(vocab: Vocabulary, prompt: str) -> list[int]:
    """Tokenize a prompt with the corpus tokenizer and encode it to IDs."""
    ids = vocab.encode(tokenize(prompt))
    if not ids:
        raise ValueError(f"prompt produced no tokens: {prompt!r}")
    return ids


def check_vocab_matches_model(vocab: Vocabulary, model: RecurrentLanguageModel) -> None:
    """Fail clearly when a checkpoint's vocabulary size disagrees with the corpus.

    A checkpoint trained on a different vocabulary is not a subtle bug: the
    output layer indexes a different token set, so every predicted word would be
    wrong. Catching it here turns an ``IndexError`` into an actionable message.
    """
    if model.config.vocab_size != vocab.size:
        raise ValueError(
            f"model was built for {model.config.vocab_size} tokens but the vocabulary "
            f"has {vocab.size}; use the vocab.json from the same processed corpus"
        )


@torch.no_grad()
def next_word_predictions(
    model: RecurrentLanguageModel,
    vocab: Vocabulary,
    prompt: str,
    *,
    top_n: int = 10,
    context_length: int = 64,
    device: str | torch.device = "cpu",
) -> list[Prediction]:
    """Return the ``top_n`` most probable next words after ``prompt``."""
    check_vocab_matches_model(vocab, model)
    model = model.to(device).eval()
    ids = tokenize_prompt(vocab, prompt)[-context_length:]
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    probabilities = torch.softmax(model(inputs)[0, -1], dim=-1)
    top_n = min(top_n, vocab.size)
    values, indices = torch.topk(probabilities, top_n)
    return [
        Prediction(token=vocab.id_to_token[int(index)], probability=float(value))
        for value, index in zip(values.tolist(), indices.tolist())
    ]


def sample_next_token(
    logits: torch.Tensor,
    *,
    greedy: bool = False,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Choose the next token from ``logits`` of shape ``[1, V]``.

    ``greedy`` (or a non-positive temperature) takes the argmax. Otherwise the
    logits are divided by the temperature and optionally restricted to the
    ``top_k`` candidates before sampling.
    """
    if greedy or temperature <= 0:
        return logits.argmax(dim=-1)

    scaled = logits / temperature
    if top_k is not None and top_k < scaled.size(-1):
        threshold = torch.topk(scaled, top_k, dim=-1).values[..., -1, None]
        scaled = torch.where(scaled < threshold, -torch.inf, scaled)
    probabilities = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate_tokens(
    model: RecurrentLanguageModel,
    vocab: Vocabulary,
    prompt: str,
    *,
    words: int = 100,
    temperature: float = 0.8,
    top_k: int | None = None,
    greedy: bool = False,
    context_length: int = 64,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> list[str]:
    """Generate ``words`` new words after ``prompt`` and return them as tokens.

    The whole context is re-fed at every step and truncated to the most recent
    ``context_length`` tokens, so the model never sees more context than it was
    trained on.
    """
    if words < 0:
        raise ValueError(f"words must be >= 0, got {words}")

    check_vocab_matches_model(vocab, model)
    model = model.to(device).eval()
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    ids = tokenize_prompt(vocab, prompt)
    generated: list[str] = []
    for _ in range(words):
        window = ids[-context_length:]
        inputs = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(inputs)[0, -1].unsqueeze(0).to("cpu")
        next_id = int(
            sample_next_token(
                logits,
                greedy=greedy,
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            ).item()
        )
        ids.append(next_id)
        generated.append(vocab.id_to_token[next_id])

    return generated


def generate(
    model: RecurrentLanguageModel,
    vocab: Vocabulary,
    prompt: str,
    *,
    words: int = 100,
    temperature: float = 0.8,
    top_k: int | None = None,
    greedy: bool = False,
    context_length: int = 64,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> str:
    """Generate ``words`` new words after ``prompt`` and render them as text."""
    generated = generate_tokens(
        model,
        vocab,
        prompt,
        words=words,
        temperature=temperature,
        top_k=top_k,
        greedy=greedy,
        context_length=context_length,
        device=device,
        seed=seed,
    )
    return join_tokens(prompt, generated)


def join_tokens(prompt: str, generated: Sequence[str]) -> str:
    """Reattach generated words to the prompt without disturbing the prompt text.

    Punctuation and possessive suffixes are glued to the preceding word, which
    is what the word-level tokenizer split apart in the first place.
    """
    if not generated:
        return prompt
    text = prompt.rstrip()
    for token in generated:
        if token.startswith("'") or token in CLOSING_PUNCTUATION:
            text += token
        else:
            text += " " + token
    return text


def format_predictions(predictions: Sequence[Prediction]) -> str:
    """Render predictions as a token/probability table."""
    width = max((len(p.token) for p in predictions), default=4)
    return "\n".join(
        f"{p.token:<{width}}  {p.probability:>7.4f}" for p in predictions
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.generate", description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--top-n", type=int, default=10, help="next-word candidates to list")
    parser.add_argument("--words", type=int, default=0, help="words to generate (0 = predict only)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--greedy", action="store_true", help="always take the most likely word")
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Imported here to keep the module importable without pulling in training.
    from src.train import load_checkpoint, resolve_device

    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    vocab = Vocabulary.load(Path(args.processed_dir) / "vocab.json")
    contents = load_checkpoint(args.checkpoint, device=device)

    print(f"prompt: {args.prompt}\n")
    predictions = next_word_predictions(
        contents.model,
        vocab,
        args.prompt,
        top_n=args.top_n,
        context_length=args.context_length,
        device=device,
    )
    print(f"top {len(predictions)} next words:")
    print(format_predictions(predictions))

    if args.words > 0:
        mode = "greedy" if args.greedy else f"temperature {args.temperature}"
        if args.top_k:
            mode += f", top-k {args.top_k}"
        text = generate(
            contents.model,
            vocab,
            args.prompt,
            words=args.words,
            temperature=args.temperature,
            top_k=args.top_k,
            greedy=args.greedy,
            context_length=args.context_length,
            device=device,
            seed=args.seed,
        )
        print(f"\ngenerated {args.words} words ({mode}):\n")
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
