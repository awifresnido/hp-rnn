"""The recurrent language model: embedding -> recurrent cell -> vocabulary logits.

One class covers all three architectures in the experiment. Switching between
them is a configuration change (``cell="rnn" | "gru" | "lstm"``), which is what
makes the Stage 10-14 comparison a controlled one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

#: Supported recurrent cell types, in the order the project compares them.
CELL_TYPES = ("rnn", "gru", "lstm")

_CELLS: dict[str, type[nn.RNNBase]] = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


@dataclass
class ModelConfig:
    """Architecture settings for :class:`RecurrentLanguageModel`."""

    vocab_size: int
    embed_size: int = 256
    hidden_size: int = 512
    num_layers: int = 2
    cell: str = "rnn"
    dropout: float = 0.0
    pad_id: int = 0

    def __post_init__(self) -> None:
        if self.cell not in CELL_TYPES:
            raise ValueError(f"cell must be one of {CELL_TYPES}, got {self.cell!r}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.vocab_size < 1 or self.embed_size < 1 or self.hidden_size < 1:
            raise ValueError("vocab_size, embed_size, and hidden_size must be positive")

    def to_dict(self) -> dict[str, object]:
        """Serialise so checkpoints describe their own architecture."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ModelConfig:
        return cls(**payload)  # type: ignore[arg-type]


class RecurrentLanguageModel(nn.Module):
    """Word-level language model: ``[B, T]`` token IDs -> ``[B, T, V]`` logits."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size, config.embed_size, padding_idx=config.pad_id
        )
        # nn.RNN/GRU/LSTM only apply dropout between stacked layers, so with a
        # single layer the setting has to be dropped rather than passed on.
        recurrent_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.recurrent = _CELLS[config.cell](
            input_size=config.embed_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output_dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map a batch of token IDs to logits over the vocabulary at every step.

        ``inputs`` has shape ``[B, T]``; the result has shape ``[B, T, V]``.
        """
        embedded = self.embedding(inputs)
        # LSTM returns (output, (h, c)) while RNN/GRU return (output, h); both
        # unpack identically, so no cell-specific branch is needed here.
        output, _hidden = self.recurrent(embedded)
        return self.head(self.output_dropout(output))

    def num_parameters(self) -> int:
        """Count trainable parameters, for the final comparison table."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
