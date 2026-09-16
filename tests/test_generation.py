"""Stages 8-9 tests: next-word probabilities, greedy, temperature, and top-k."""

from __future__ import annotations

import math

import pytest
import torch

from src.data import Vocabulary
from src.generate import (
    Prediction,
    generate,
    generate_tokens,
    next_word_predictions,
    sample_next_token,
    tokenize_prompt,
)
from src.model import ModelConfig, RecurrentLanguageModel

VOCAB_SIZE = 32


@pytest.fixture
def vocab() -> Vocabulary:
    return Vocabulary.build(
        ["Harry", "looked", "at", "Ron", ".", "Hermione", "and", "smiled"],
        max_size=VOCAB_SIZE,
    )


@pytest.fixture
def model(vocab: Vocabulary) -> RecurrentLanguageModel:
    """A model whose output layer matches the fixture vocabulary exactly."""
    torch.manual_seed(0)
    return RecurrentLanguageModel(
        ModelConfig(vocab_size=vocab.size, embed_size=8, hidden_size=8, num_layers=1)
    ).eval()


def test_tokenize_prompt_reuses_the_corpus_tokenizer(vocab: Vocabulary) -> None:
    assert tokenize_prompt(vocab, "Harry looked at Ron.") == vocab.encode(
        ["Harry", "looked", "at", "Ron", "."]
    )


def test_predictions_are_ranked_and_sum_to_at_most_one(
    model: RecurrentLanguageModel, vocab: Vocabulary
) -> None:
    predictions = next_word_predictions(model, vocab, "Harry looked at", top_n=5)

    assert len(predictions) == 5
    assert all(isinstance(p, Prediction) for p in predictions)
    probabilities = [p.probability for p in predictions]
    assert probabilities == sorted(probabilities, reverse=True)
    assert 0.0 < sum(probabilities) <= 1.0 + 1e-6
    assert all(p.token != "" for p in predictions)


def test_top_n_is_capped_by_vocabulary_size(
    model: RecurrentLanguageModel, vocab: Vocabulary
) -> None:
    predictions = next_word_predictions(model, vocab, "Harry", top_n=1000)

    assert len(predictions) == vocab.size


def test_greedy_selection_picks_the_highest_logit() -> None:
    logits = torch.tensor([[0.1, 5.0, 0.2]])

    chosen = sample_next_token(logits, greedy=True, temperature=1.0, top_k=None, generator=None)

    assert int(chosen.item()) == 1


def test_top_k_of_one_equals_greedy() -> None:
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(1, VOCAB_SIZE, generator=torch.Generator().manual_seed(1))

    greedy = sample_next_token(logits, greedy=True, temperature=1.0, top_k=None, generator=None)
    top_one = sample_next_token(
        logits, greedy=False, temperature=1.0, top_k=1, generator=generator
    )

    assert int(greedy.item()) == int(top_one.item()) == int(logits.argmax(dim=-1).item())


def test_zero_temperature_falls_back_to_greedy() -> None:
    logits = torch.randn(1, VOCAB_SIZE, generator=torch.Generator().manual_seed(2))

    chosen = sample_next_token(logits, greedy=False, temperature=0.0, top_k=None, generator=None)

    assert int(chosen.item()) == int(logits.argmax(dim=-1).item())


def test_generation_is_reproducible_with_a_seed(
    model: RecurrentLanguageModel, vocab: Vocabulary
) -> None:
    first = generate(model, vocab, "Harry looked at", words=15, temperature=1.0, seed=7)
    second = generate(model, vocab, "Harry looked at", words=15, temperature=1.0, seed=7)

    assert first == second


def test_generation_produces_the_requested_number_of_words(
    model: RecurrentLanguageModel, vocab: Vocabulary
) -> None:
    tokens = generate_tokens(model, vocab, "Harry looked at", words=10, greedy=True)

    assert len(tokens) == 10
    assert all(token in vocab.token_to_id for token in tokens)


def test_rendered_generation_keeps_the_prompt_intact(
    model: RecurrentLanguageModel, vocab: Vocabulary
) -> None:
    prompt = "Harry looked at"
    text = generate(model, vocab, prompt, words=10, greedy=True)

    assert text.startswith(prompt)
    assert len(text) > len(prompt)


def test_only_the_last_context_tokens_influence_generation(
    model: RecurrentLanguageModel, vocab: Vocabulary
) -> None:
    base = ["Harry", "looked", "at", "Ron", "and", "Hermione"]
    long_prompt = " ".join(base * 10)
    tail_prompt = " ".join((base * 10)[-16:])

    from_long = generate_tokens(
        model, vocab, long_prompt, words=1, greedy=True, context_length=16
    )
    from_tail = generate_tokens(
        model, vocab, tail_prompt, words=1, greedy=True, context_length=16
    )

    assert from_long == from_tail


def test_temperature_affects_the_sampling_distribution(
    model: RecurrentLanguageModel, vocab: Vocabulary
) -> None:
    low = generate(model, vocab, "Harry", words=40, temperature=0.2, seed=3)
    high = generate(model, vocab, "Harry", words=40, temperature=1.5, seed=3)

    assert low != high


def test_mismatched_vocabulary_and_model_are_rejected(
    model: RecurrentLanguageModel,
) -> None:
    other = Vocabulary.build(["Harry", "Ron", "Hermione", "wand", "cloak"])

    with pytest.raises(ValueError, match="vocabulary"):
        next_word_predictions(model, other, "Harry")


def test_prediction_probabilities_are_finite(
    model: RecurrentLanguageModel, vocab: Vocabulary
) -> None:
    predictions = next_word_predictions(model, vocab, "Harry", top_n=10)

    assert all(math.isfinite(p.probability) for p in predictions)
    assert all(0.0 <= p.probability <= 1.0 for p in predictions)
