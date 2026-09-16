"""Independent evaluation: split losses and perplexity from a saved checkpoint.

This module never imports :mod:`src.train`, so a checkpoint can be evaluated
without the training code being involved.

Perplexity is ``exp(cross entropy)``: the effective number of equally likely
words the model is choosing between at each step.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import build_dataloaders, build_split_dataloaders
from src.model import ModelConfig

#: The three contiguous corpus sections produced by :mod:`src.data`.
SPLITS = ("train", "val", "test")


def sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy over every predicted position in a ``[B, T, V]`` batch."""
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
    )


def perplexity_from_loss(loss: float) -> float:
    """Convert a cross-entropy loss in nats into perplexity."""
    return math.exp(loss)


@dataclass(frozen=True)
class EvaluationReport:
    """Loss, perplexity, and model size for one checkpoint."""

    split_losses: dict[str, float]
    split_perplexities: dict[str, float]
    model_config: ModelConfig
    num_parameters: int
    checkpoint_path: Path | None = None

    def table(self) -> str:
        """Render the report as the comparison table the project asks for."""
        lines = [
            f"{'model':<14}{'params':>12}{'loss':>10}{'perplexity':>12}",
            "-" * 48,
        ]
        for split in SPLITS:
            if split not in self.split_losses:
                continue
            lines.append(
                f"{self.model_config.cell + ' ' + split:<14}"
                f"{self.num_parameters:>12,}"
                f"{self.split_losses[split]:>10.4f}"
                f"{self.split_perplexities[split]:>12.2f}"
            )
        return "\n".join(lines)


@torch.no_grad()
def split_losses(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Average cross-entropy loss over each split, with the model in eval mode."""
    device = torch.device(device)
    model.to(device).eval()
    losses: dict[str, float] = {}
    for split, loader in loaders.items():
        total_loss = 0.0
        total_positions = 0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            batch_positions = targets.numel()
            total_loss += float(sequence_cross_entropy(logits, targets)) * batch_positions
            total_positions += batch_positions
        if total_positions == 0:
            raise ValueError(f"split {split!r} produced no batches to evaluate")
        losses[split] = total_loss / total_positions
    return losses


def evaluate_checkpoint(
    checkpoint_path: Path,
    processed_dir: Path,
    device: str = "auto",
    batch_size: int = 128,
    sequence_length: int | None = None,
) -> EvaluationReport:
    """Evaluate a saved checkpoint against the prepared corpus splits."""
    from src.train import load_checkpoint, load_processed, resolve_device

    checkpoint_path = Path(checkpoint_path)
    contents = load_checkpoint(checkpoint_path)
    target_device = resolve_device(device)

    processed = load_processed(processed_dir)
    length = sequence_length or contents.sequence_length or processed.sequence_length
    if processed.splits:
        loaders = build_split_dataloaders(processed.splits, length, batch_size)
    else:
        loaders = build_dataloaders(processed.token_ids, length, batch_size)

    losses = split_losses(contents.model, loaders, device=target_device)
    return EvaluationReport(
        split_losses=losses,
        split_perplexities={split: perplexity_from_loss(loss) for split, loss in losses.items()},
        model_config=contents.model_config,
        num_parameters=contents.model.num_parameters(),
        checkpoint_path=checkpoint_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.evaluate", description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = evaluate_checkpoint(
        args.checkpoint,
        args.processed_dir,
        device=args.device,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )
    print(f"checkpoint : {report.checkpoint_path}")
    print(f"cell       : {report.model_config.cell}")
    print(f"parameters : {report.num_parameters:,}\n")
    print(report.table())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
