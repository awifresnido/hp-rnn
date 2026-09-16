"""Stage 7 tests: independent evaluation, loss, and perplexity."""

from __future__ import annotations

import math

import pytest
import torch

from src.data import build_dataloaders
from src.evaluate import (
    evaluate_checkpoint,
    perplexity_from_loss,
    sequence_cross_entropy,
    split_losses,
)
from src.model import ModelConfig, RecurrentLanguageModel
from src.train import save_checkpoint

VOCAB_SIZE = 40
SEQUENCE_LENGTH = 12


def test_perplexity_is_the_exponential_of_cross_entropy() -> None:
    assert perplexity_from_loss(0.0) == pytest.approx(1.0)
    assert perplexity_from_loss(math.log(10.0)) == pytest.approx(10.0)
    assert perplexity_from_loss(math.log(VOCAB_SIZE)) == pytest.approx(VOCAB_SIZE)


def test_sequence_cross_entropy_matches_manual_flattened_computation() -> None:
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 5, VOCAB_SIZE, generator=generator)
    targets = torch.randint(0, VOCAB_SIZE, (2, 5), generator=generator)

    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

    assert torch.allclose(sequence_cross_entropy(logits, targets), expected)


def test_split_losses_reports_a_loss_for_every_split() -> None:
    ids = torch.randint(0, VOCAB_SIZE, (500,), generator=torch.Generator().manual_seed(2)).tolist()
    loaders = build_dataloaders(ids, SEQUENCE_LENGTH, batch_size=16)
    model = RecurrentLanguageModel(
        ModelConfig(vocab_size=VOCAB_SIZE, embed_size=8, hidden_size=8, num_layers=1)
    )

    losses = split_losses(model, loaders, device="cpu")

    assert set(losses) == {"train", "val", "test"}
    assert all(math.isfinite(value) and value > 0 for value in losses.values())


def test_evaluate_checkpoint_works_without_any_training(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    ids = torch.randint(0, VOCAB_SIZE, (500,), generator=torch.Generator().manual_seed(3))
    torch.save(
        {
            "token_ids": ids,
            "sequence_length": SEQUENCE_LENGTH,
            "vocab_size": VOCAB_SIZE,
        },
        processed_dir / "token_ids.pt",
    )

    model = RecurrentLanguageModel(
        ModelConfig(vocab_size=VOCAB_SIZE, embed_size=8, hidden_size=8, num_layers=1, cell="gru")
    )
    checkpoint = tmp_path / "gru-best.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        model_config=model.config,
        sequence_length=SEQUENCE_LENGTH,
        epoch=1,
        val_loss=2.0,
        step=10,
    )

    report = evaluate_checkpoint(checkpoint, processed_dir, device="cpu", batch_size=16)

    assert set(report.split_losses) == {"train", "val", "test"}
    assert report.split_perplexities["test"] == pytest.approx(
        math.exp(report.split_losses["test"])
    )
    assert report.num_parameters == model.num_parameters()
    assert report.model_config.cell == "gru"
