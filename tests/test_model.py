"""Stage 5 tests: recurrent language model shapes, gradients, and configuration."""

from __future__ import annotations

import pytest
import torch

from src.model import CELL_TYPES, ModelConfig, RecurrentLanguageModel

BATCH_SIZE = 4
SEQUENCE_LENGTH = 64
VOCAB_SIZE = 128


def build_model(cell: str = "rnn", **overrides) -> RecurrentLanguageModel:
    settings = {
        "vocab_size": VOCAB_SIZE,
        "embed_size": 32,
        "hidden_size": 48,
        "num_layers": 2,
        "cell": cell,
    }
    settings.update(overrides)
    return RecurrentLanguageModel(ModelConfig(**settings))


def sample_inputs() -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH))


@pytest.mark.parametrize("cell", CELL_TYPES)
def test_model_maps_batch_and_time_to_vocabulary_logits(cell: str) -> None:
    logits = build_model(cell)(sample_inputs())

    assert logits.shape == (BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE)


@pytest.mark.parametrize("cell", CELL_TYPES)
def test_backward_produces_gradients_for_every_parameter(cell: str) -> None:
    model = build_model(cell)
    logits = model(sample_inputs())
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH))

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
    )
    loss.backward()

    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []
    assert model.embedding.weight.grad.abs().sum().item() > 0
    assert model.head.weight.grad.abs().sum().item() > 0


@pytest.mark.parametrize("cell", CELL_TYPES)
def test_parameter_count_is_positive(cell: str) -> None:
    assert build_model(cell).num_parameters() > 0


def test_parameter_count_grows_with_hidden_size() -> None:
    small = build_model(hidden_size=16).num_parameters()
    large = build_model(hidden_size=64).num_parameters()

    assert large > small


def test_dropout_keeps_output_shape_valid() -> None:
    logits = build_model("lstm", dropout=0.4)(sample_inputs())

    assert logits.shape == (BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE)
    assert torch.isfinite(logits).all()


def test_eval_mode_is_deterministic() -> None:
    model = build_model("gru", dropout=0.4).eval()
    inputs = sample_inputs()

    with torch.no_grad():
        first = model(inputs)
        second = model(inputs)

    assert torch.equal(first, second)


def test_config_round_trips_through_dict() -> None:
    config = ModelConfig(vocab_size=100, embed_size=16, hidden_size=32, num_layers=1, cell="lstm")

    restored = ModelConfig.from_dict(config.to_dict())

    assert restored == config


def test_unknown_cell_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="cell"):
        build_model("transformer")
