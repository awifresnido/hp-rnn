"""Stage 6 tests: one-batch overfit, gradient clipping, and checkpoint round-trip."""

from __future__ import annotations

import torch

from src.model import ModelConfig
from src.train import (
    TrainConfig,
    clip_gradients_,
    load_checkpoint,
    overfit_one_batch,
    save_checkpoint,
    train_model,
)

VOCAB_SIZE = 48
SEQUENCE_LENGTH = 16


def tiny_model(cell: str = "rnn") -> torch.nn.Module:
    from src.model import RecurrentLanguageModel

    return RecurrentLanguageModel(
        ModelConfig(
            vocab_size=VOCAB_SIZE,
            embed_size=16,
            hidden_size=16,
            num_layers=1,
            cell=cell,
        )
    )


def tiny_batch(size: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randint(0, VOCAB_SIZE, (size, SEQUENCE_LENGTH), generator=generator)
    targets = torch.randint(0, VOCAB_SIZE, (size, SEQUENCE_LENGTH), generator=generator)
    return inputs, targets


def test_overfit_one_batch_reduces_loss_substantially() -> None:
    inputs, targets = tiny_batch()

    history = overfit_one_batch(
        tiny_model(), inputs, targets, steps=120, learning_rate=0.02, seed=0
    )

    assert history[-1] < history[0] * 0.5
    assert history[-1] < 1.0


def test_overfit_one_batch_is_reproducible_with_a_seed() -> None:
    inputs, targets = tiny_batch()

    # Model initialisation must be seeded too, not just the optimization run.
    torch.manual_seed(42)
    first = overfit_one_batch(tiny_model(), inputs, targets, steps=10, seed=42)

    torch.manual_seed(42)
    second = overfit_one_batch(tiny_model(), inputs, targets, steps=10, seed=42)

    assert first == second


def test_clip_gradients_bounds_the_total_norm() -> None:
    model = tiny_model()
    inputs, targets = tiny_batch()
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
    )
    loss.backward()

    pre_clip_norm = clip_gradients_(model, max_norm=0.05)

    total = torch.sqrt(
        sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)
    )
    assert pre_clip_norm > 0.05
    assert total.item() <= 0.05 + 1e-5


def test_checkpoint_round_trip_reproduces_predictions(tmp_path) -> None:
    torch.manual_seed(0)
    model = tiny_model("gru").eval()
    inputs, _ = tiny_batch()
    path = tmp_path / "gru-best.pt"

    save_checkpoint(
        path,
        model=model,
        model_config=model.config,
        sequence_length=SEQUENCE_LENGTH,
        epoch=3,
        val_loss=1.234,
        step=100,
    )
    loaded = load_checkpoint(path)

    with torch.no_grad():
        before = model(inputs)
        after = loaded.model(inputs)

    assert torch.allclose(before, after, atol=1e-6)


def test_checkpoint_records_metadata_for_standalone_evaluation(tmp_path) -> None:
    model = tiny_model("lstm")
    path = tmp_path / "lstm-best.pt"

    save_checkpoint(
        path,
        model=model,
        model_config=model.config,
        sequence_length=32,
        epoch=7,
        val_loss=4.25,
        step=512,
    )
    loaded = load_checkpoint(path)

    assert loaded.epoch == 7
    assert loaded.val_loss == 4.25
    assert loaded.sequence_length == 32
    assert loaded.model_config.cell == "lstm"
    assert loaded.model_config.vocab_size == VOCAB_SIZE


def test_train_model_saves_best_checkpoint_and_tracks_history(tmp_path) -> None:
    ids = torch.randint(0, VOCAB_SIZE, (600,), generator=torch.Generator().manual_seed(1))
    # Mirror what src.data.prepare_dataset writes, so training reads the same contract.
    torch.save(
        {
            "token_ids": ids,
            "sequence_length": SEQUENCE_LENGTH,
            "vocab_size": VOCAB_SIZE,
        },
        tmp_path / "token_ids.pt",
    )

    config = TrainConfig(
        processed_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoints",
        run_name="rnn-smoke",
        cell="rnn",
        embed_size=16,
        hidden_size=16,
        num_layers=1,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=8,
        epochs=2,
        learning_rate=0.01,
        device="cpu",
        seed=0,
    )

    result = train_model(config)

    assert (tmp_path / "checkpoints" / "rnn-smoke-best.pt").exists()
    assert len(result.history) == 2
    assert result.best_val_loss == min(epoch.val_loss for epoch in result.history)
    assert result.best_val_loss > 0
