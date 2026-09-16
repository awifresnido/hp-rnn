"""Stage 14 tests: aggregating runs into the model comparison table."""

from __future__ import annotations

import json
import math

from src.compare import RunSummary, format_table, load_run
from src.model import ModelConfig, RecurrentLanguageModel
from src.train import save_checkpoint

VOCAB_SIZE = 40


def write_run(
    results_dir,
    checkpoint_dir,
    run_name: str,
    cell: str,
    *,
    val_loss: float = 2.0,
    epochs: int = 2,
    seconds: float = 100.0,
) -> None:
    """Write the history JSON and best checkpoint that a finished run leaves behind."""
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = RecurrentLanguageModel(
        ModelConfig(vocab_size=VOCAB_SIZE, embed_size=8, hidden_size=8, num_layers=1, cell=cell)
    )
    checkpoint = checkpoint_dir / f"{run_name}-best.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        model_config=model.config,
        sequence_length=16,
        epoch=1,
        val_loss=val_loss,
        step=10,
        run_name=run_name,
    )
    (results_dir / f"{run_name}-history.json").write_text(
        json.dumps(
            {
                "train_config": {"cell": cell, "run_name": run_name, "seed": 0},
                "history": [
                    {
                        "epoch": number,
                        "train_loss": val_loss + 1.0,
                        "val_loss": val_loss,
                        "seconds": seconds,
                    }
                    for number in range(1, epochs + 1)
                ],
                "best_val_loss": val_loss,
                "best_checkpoint": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )


def test_load_run_reads_metrics_from_history_and_checkpoint(tmp_path) -> None:
    results = tmp_path / "results"
    checkpoints = tmp_path / "checkpoints"
    write_run(results, checkpoints, "lstm", "lstm", val_loss=2.0, epochs=3, seconds=120.0)

    summary = load_run("lstm", results, checkpoints)

    assert isinstance(summary, RunSummary)
    assert summary.cell == "lstm"
    assert summary.epochs == 3
    assert summary.best_val_loss == 2.0
    assert summary.val_perplexity > 0
    assert summary.training_seconds == 360.0
    assert summary.parameters > 0
    assert summary.test_perplexity is None


def test_val_perplexity_is_exp_of_best_val_loss(tmp_path) -> None:
    results = tmp_path / "results"
    checkpoints = tmp_path / "checkpoints"
    write_run(results, checkpoints, "gru", "gru", val_loss=math.log(50.0))

    summary = load_run("gru", results, checkpoints)

    assert summary.val_perplexity == round(math.exp(math.log(50.0)), 4)


def test_table_lists_every_run_in_the_requested_order(tmp_path) -> None:
    results = tmp_path / "results"
    checkpoints = tmp_path / "checkpoints"
    write_run(results, checkpoints, "rnn", "rnn")
    write_run(results, checkpoints, "lstm", "lstm")

    summaries = [load_run("rnn", results, checkpoints), load_run("lstm", results, checkpoints)]
    table = format_table(summaries)

    assert "rnn" in table and "lstm" in table
    assert "Parameters" in table
    assert "Test PPL" in table
    assert table.index("rnn") < table.index("lstm")


def test_unevaluated_test_perplexity_is_shown_as_pending(tmp_path) -> None:
    results = tmp_path / "results"
    checkpoints = tmp_path / "checkpoints"
    write_run(results, checkpoints, "rnn", "rnn")

    table = format_table([load_run("rnn", results, checkpoints)])

    assert "pending" in table


def test_missing_metrics_render_as_not_available(tmp_path) -> None:
    summary = RunSummary(
        run_name="broken",
        cell="rnn",
        parameters=None,
        best_val_loss=None,
        val_perplexity=None,
        test_perplexity=None,
        training_seconds=None,
        epochs=None,
        best_epoch=None,
    )

    table = format_table([summary])

    assert "n/a" in table


def test_empty_run_list_produces_a_header_only_table() -> None:
    table = format_table([])

    assert "Parameters" in table
    assert len(table.splitlines()) == 2


def test_missing_history_file_is_reported_clearly(tmp_path) -> None:
    try:
        load_run("ghost", tmp_path / "results", tmp_path / "checkpoints")
    except FileNotFoundError as error:
        assert "ghost" in str(error)
    else:  # pragma: no cover - the call above must raise
        raise AssertionError("expected FileNotFoundError for a missing run")
