"""Training: the one-batch diagnostic, gradient clipping, and the training loop.

The one-batch overfit check runs first on purpose. If the model cannot memorize
a single small batch, the implementation is wrong and longer training on the
whole corpus would only hide the bug behind a slow loss curve.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from src.evaluate import sequence_cross_entropy, split_losses
from src.model import CELL_TYPES, ModelConfig, RecurrentLanguageModel


def resolve_device(requested: str = "auto") -> torch.device:
    """Turn a device request into a device, refusing a silent CPU fallback."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device {requested!r} was requested but CUDA is not available"
        )
    return device


@dataclass
class TrainConfig:
    """Everything that affects a training run, so runs stay comparable."""

    processed_dir: Path = Path("data/processed")
    checkpoint_dir: Path = Path("checkpoints")
    run_name: str = "rnn"
    cell: str = "rnn"
    embed_size: int = 256
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.0
    sequence_length: int = 64
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 15
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 0
    overfit_only: bool = False

    def model_config(self, vocab_size: int) -> ModelConfig:
        return ModelConfig(
            vocab_size=vocab_size,
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            cell=self.cell,
            dropout=self.dropout,
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["processed_dir"] = str(self.processed_dir)
        payload["checkpoint_dir"] = str(self.checkpoint_dir)
        return payload


@dataclass(frozen=True)
class EpochRecord:
    """One row of the training history."""

    epoch: int
    train_loss: float
    val_loss: float
    seconds: float


@dataclass
class TrainResult:
    """Outcome of :func:`train_model`."""

    history: list[EpochRecord] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_checkpoint: Path | None = None
    overfit_history: list[float] = field(default_factory=list)


@dataclass
class CheckpointContents:
    """A checkpoint loaded back into a usable model plus its metadata."""

    model: RecurrentLanguageModel
    model_config: ModelConfig
    sequence_length: int
    epoch: int
    val_loss: float
    step: int
    run_name: str


def clip_gradients_(model: nn.Module, max_norm: float) -> float:
    """Clip gradients in place and return the total norm measured beforehand."""
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.0
    return float(nn.utils.clip_grad_norm_(parameters, max_norm))


def set_seed(seed: int) -> None:
    """Seed Python and torch so a run can be repeated."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    *,
    model: RecurrentLanguageModel,
    model_config: ModelConfig,
    sequence_length: int,
    epoch: int,
    val_loss: float,
    step: int = 0,
    run_name: str = "",
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    """Write a self-describing checkpoint (architecture + progress in one file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config.to_dict(),
            "sequence_length": sequence_length,
            "epoch": epoch,
            "val_loss": val_loss,
            "step": step,
            "run_name": run_name,
            "optimizer_state": optimizer.state_dict() if optimizer else None,
        },
        path,
    )
    return path


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> CheckpointContents:
    """Rebuild a model from a checkpoint, evaluating on ``device``."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model_config = ModelConfig.from_dict(payload["model_config"])
    model = RecurrentLanguageModel(model_config)
    model.load_state_dict(payload["model_state"])
    model.to(torch.device(device)).eval()
    return CheckpointContents(
        model=model,
        model_config=model_config,
        sequence_length=int(payload.get("sequence_length", 0)),
        epoch=int(payload.get("epoch", 0)),
        val_loss=float(payload.get("val_loss", float("nan"))),
        step=int(payload.get("step", 0)),
        run_name=str(payload.get("run_name", "")),
    )


def overfit_one_batch(
    model: RecurrentLanguageModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int = 200,
    learning_rate: float = 0.01,
    grad_clip: float = 1.0,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> list[float]:
    """Train repeatedly on one batch and return the loss after each step.

    This is the pre-flight check: a correct implementation must drive the loss
    down sharply on a single small batch.
    """
    if seed is not None:
        set_seed(seed)
    device = torch.device(device)
    model.to(device).train()
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = sequence_cross_entropy(model(inputs), targets)
        loss.backward()
        clip_gradients_(model, grad_clip)
        optimizer.step()
        history.append(loss.detach().item())
    return history


def load_processed(processed_dir: Path) -> tuple[list[int], int, int]:
    """Read token IDs, sequence length, and vocabulary size from ``processed_dir``."""
    processed_dir = Path(processed_dir)
    payload = torch.load(processed_dir / "token_ids.pt", weights_only=False)
    vocab_size = payload.get("vocab_size")
    if vocab_size is None:
        payload_vocab = json.loads((processed_dir / "vocab.json").read_text(encoding="utf-8"))
        vocab_size = len(payload_vocab["id_to_token"])
    return payload["token_ids"].tolist(), int(payload["sequence_length"]), int(vocab_size)


def train_model(config: TrainConfig) -> TrainResult:
    """Run the one-batch check (optionally) and train, keeping the best epoch."""
    from src.data import build_dataloaders

    set_seed(config.seed)
    device = resolve_device(config.device)
    token_ids, stored_length, vocab_size = load_processed(config.processed_dir)
    sequence_length = config.sequence_length or stored_length
    loaders = build_dataloaders(token_ids, sequence_length, config.batch_size)

    model = RecurrentLanguageModel(config.model_config(vocab_size)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    result = TrainResult()

    if config.overfit_only:
        inputs, targets = next(iter(loaders["train"]))
        result.overfit_history = overfit_one_batch(
            model,
            inputs,
            targets,
            steps=200,
            learning_rate=config.learning_rate,
            grad_clip=config.grad_clip,
            device=device,
        )
        return result

    checkpoint_path = Path(config.checkpoint_dir) / f"{config.run_name}-best.pt"
    global_step = 0
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        running_loss = 0.0
        running_positions = 0
        for inputs, targets in loaders["train"]:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = sequence_cross_entropy(model(inputs), targets)
            loss.backward()
            clip_gradients_(model, config.grad_clip)
            optimizer.step()
            running_loss += loss.detach().item() * targets.numel()
            running_positions += targets.numel()
            global_step += 1

        train_loss = running_loss / running_positions
        val_loss = split_losses(model, {"val": loaders["val"]}, device=str(device))["val"]
        result.history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                seconds=time.perf_counter() - started,
            )
        )

        if val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            result.best_checkpoint = save_checkpoint(
                checkpoint_path,
                model=model,
                model_config=model.config,
                sequence_length=sequence_length,
                epoch=epoch,
                val_loss=val_loss,
                step=global_step,
                run_name=config.run_name,
                optimizer=optimizer,
            )
    return result


def format_history(result: TrainResult) -> str:
    """Render the per-epoch table the training loop prints."""
    lines = [f"{'epoch':>6}{'train loss':>12}{'val loss':>12}{'seconds':>10}", "-" * 40]
    for record in result.history:
        lines.append(
            f"{record.epoch:>6}{record.train_loss:>12.4f}"
            f"{record.val_loss:>12.4f}{record.seconds:>10.1f}"
        )
    if result.best_checkpoint:
        lines.append(f"\nbest val loss {result.best_val_loss:.4f} -> {result.best_checkpoint}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.train", description=__doc__)
    parser.add_argument("--model", dest="cell", choices=CELL_TYPES, default="rnn")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--embed-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overfit-one-batch",
        action="store_true",
        help="run only the one-batch memorisation check and stop",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = TrainConfig(
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name or args.cell,
        cell=args.cell,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        device=args.device,
        seed=args.seed,
        overfit_only=args.overfit_one_batch,
    )
    print(f"device: {resolve_device(config.device)}")
    result = train_model(config)

    if config.overfit_only:
        history = result.overfit_history
        print(f"one-batch loss: first {history[0]:.4f} -> last {history[-1]:.4f}")
        print(f"reduction: {100 * (1 - history[-1] / history[0]):.1f}%")
        return 0

    print(format_history(result))
    history_path = Path(args.results_dir) / f"{config.run_name}-history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(
            {
                "train_config": config.to_dict(),
                "history": [asdict(record) for record in result.history],
                "best_val_loss": result.best_val_loss,
                "best_checkpoint": str(result.best_checkpoint),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"wrote {history_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
