"""Aggregate finished runs into the Stage 14 model comparison.

Reads what each training run left behind — a history JSON and a best
checkpoint — so the comparison table is derived from recorded results rather
than retyped by hand.

Validation perplexity selects the model; test perplexity is reported from the
same frozen checkpoint, because a test number used for selection stops being a
test number.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

#: Order cells the way the project compares them, so tables read consistently.
CELL_ORDER = ("rnn", "gru", "lstm")

NA = "n/a"


@dataclass(frozen=True)
class RunSummary:
    """One row of the comparison table."""

    run_name: str
    cell: str
    parameters: int | None = None
    best_val_loss: float | None = None
    val_perplexity: float | None = None
    test_perplexity: float | None = None
    training_seconds: float | None = None
    epochs: int | None = None
    best_epoch: int | None = None

    def sort_key(self) -> tuple[int, str]:
        try:
            cell_rank = CELL_ORDER.index(self.cell)
        except ValueError:
            cell_rank = len(CELL_ORDER)
        return (cell_rank, self.run_name)


def history_path(run_name: str, results_dir: Path) -> Path:
    return Path(results_dir) / f"{run_name}-history.json"


def discover_runs(results_dir: Path) -> list[str]:
    """Find finished runs, ordered by architecture then name."""
    names = sorted(
        path.name[: -len("-history.json")]
        for path in Path(results_dir).glob("*-history.json")
    )
    return names


def load_run(run_name: str, results_dir: Path, checkpoint_dir: Path) -> RunSummary:
    """Build one summary from a run's history JSON and best checkpoint."""
    path = history_path(run_name, results_dir)
    if not path.exists():
        raise FileNotFoundError(f"no history for run {run_name!r} at {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history", [])
    train_config = payload.get("train_config", {})

    best_val_loss = payload.get("best_val_loss")
    if best_val_loss is None and history:
        best_val_loss = min(record["val_loss"] for record in history)

    best_epoch = None
    if history:
        best_epoch = min(history, key=lambda record: record["val_loss"])["epoch"]

    total_seconds = None
    if history:
        total_seconds = float(sum(record.get("seconds", 0.0) for record in history))

    parameters = None
    checkpoint = Path(checkpoint_dir) / f"{run_name}-best.pt"
    if checkpoint.exists():
        from src.train import load_checkpoint

        parameters = load_checkpoint(checkpoint).model.num_parameters()

    return RunSummary(
        run_name=run_name,
        cell=str(train_config.get("cell", "")),
        parameters=parameters,
        best_val_loss=best_val_loss,
        val_perplexity=None if best_val_loss is None else round(math.exp(best_val_loss), 4),
        test_perplexity=None,
        training_seconds=total_seconds,
        epochs=len(history) or None,
        best_epoch=best_epoch,
    )


def format_table(summaries: Iterable[RunSummary]) -> str:
    """Render the comparison table; unevaluated test perplexity shows as pending."""
    lines = [
        f"{'Model':<22}{'Parameters':>13}{'Val PPL':>10}{'Test PPL':>11}{'Time':>12}",
        "-" * 68,
    ]
    for summary in summaries:
        parameters = NA if summary.parameters is None else f"{summary.parameters:,}"
        val_ppl = NA if summary.val_perplexity is None else f"{summary.val_perplexity:.2f}"
        test_ppl = "pending" if summary.test_perplexity is None else f"{summary.test_perplexity:.2f}"
        if summary.training_seconds is None:
            time_cell = NA
        else:
            time_cell = f"{summary.training_seconds / 60.0:.1f} min"
        lines.append(
            f"{summary.run_name:<22}{parameters:>13}{val_ppl:>10}{test_ppl:>11}{time_cell:>12}"
        )
    return "\n".join(lines)


def to_markdown(summaries: Sequence[RunSummary]) -> str:
    """Render the same results as a markdown table for the README/report."""
    lines = [
        "| Model | Parameters | Val PPL | Test PPL | Training Time | Best Epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        test_ppl = "pending" if summary.test_perplexity is None else f"{summary.test_perplexity:.2f}"
        lines.append(
            f"| {summary.run_name} "
            f"| {NA if summary.parameters is None else f'{summary.parameters:,}'} "
            f"| {NA if summary.val_perplexity is None else f'{summary.val_perplexity:.2f}'} "
            f"| {test_ppl} "
            f"| {NA if summary.training_seconds is None else f'{summary.training_seconds / 60.0:.1f} min'} "
            f"| {NA if summary.best_epoch is None else summary.best_epoch} |"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.compare", description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--runs", nargs="*", default=None, help="run names (default: all found)")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="score each best checkpoint on the test split (touch the test set once)",
    )
    parser.add_argument("--markdown", type=Path, default=None, help="also write a markdown table")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    names = args.runs or discover_runs(args.results_dir)
    if not names:
        print(f"no runs found in {args.results_dir}")
        return 1

    summaries = sorted(
        (load_run(name, args.results_dir, args.checkpoint_dir) for name in names),
        key=RunSummary.sort_key,
    )

    if args.evaluate_test:
        from src.evaluate import evaluate_checkpoint

        scored = []
        for summary in summaries:
            checkpoint = Path(args.checkpoint_dir) / f"{summary.run_name}-best.pt"
            report = evaluate_checkpoint(
                checkpoint,
                args.processed_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
            scored.append(
                replace(summary, test_perplexity=round(report.split_perplexities["test"], 4))
            )
        summaries = scored

    print(format_table(summaries))

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(to_markdown(summaries) + "\n", encoding="utf-8")
        print(f"\nwrote {args.markdown}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
