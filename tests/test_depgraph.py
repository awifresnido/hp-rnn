"""Tests for the dependency-graph resolver.

The resolver is the part of the tooling that is easy to get subtly wrong: an
earlier version matched a call's *trailing name*, which linked a bare
``model.eval()`` inside ``src/generate.py`` to a test helper that happened to be
called ``model``. These tests pin the intended behaviour: variables never resolve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from depgraph import Function, resolve_target


def make(module: str, name: str, owner: str | None = None) -> Function:
    return Function(
        module=module,
        name=name,
        qualname=f"{owner}.{name}" if owner else name,
        kind="method" if owner else "function",
        owner=owner,
        lineno=1,
    )


@pytest.fixture
def index() -> dict[str, list[Function]]:
    entries = [
        make("src.train", "train_model"),
        make("src.data", "build_split_dataloaders"),
        make("src.model", "RecurrentLanguageModel"),
        make("src.model", "forward", owner="RecurrentLanguageModel"),
        make("src.model", "num_parameters", owner="RecurrentLanguageModel"),
        # A test helper whose name collides with a common variable name.
        make("tests.test_generation", "model"),
    ]
    index: dict[str, list[Function]] = {}
    for entry in entries:
        index.setdefault(entry.name, []).append(entry)
        index.setdefault(entry.qualname, []).append(entry)
    return index


def test_calls_on_lowercase_variables_never_resolve(index) -> None:
    caller = make("src.generate", "generate_tokens")

    assert resolve_target("model.eval", caller, index, set()) is None
    assert resolve_target("vocab.encode", caller, index, set()) is None
    assert resolve_target("loss.backward", caller, index, set()) is None


def test_bare_name_resolves_within_the_same_module(index) -> None:
    caller = make("src.train", "main")

    resolved = resolve_target("train_model", caller, index, set())

    assert resolved is not None
    assert resolved.module == "src.train"


def test_bare_name_resolves_through_an_internal_import(index) -> None:
    caller = make("src.train", "train_model")

    resolved = resolve_target(
        "build_split_dataloaders", caller, index, {"src.data", "src.data.build_split_dataloaders"}
    )

    assert resolved is not None
    assert resolved.module == "src.data"


def test_bare_name_does_not_cross_into_an_unimported_module(index) -> None:
    caller = make("src.train", "train_model")

    assert resolve_target("build_split_dataloaders", caller, index, set()) is None


def test_self_method_resolves_to_the_owning_class(index) -> None:
    caller = make("src.model", "forward", owner="RecurrentLanguageModel")

    resolved = resolve_target("self.num_parameters", caller, index, set())

    assert resolved is not None
    assert resolved.qualname == "RecurrentLanguageModel.num_parameters"


def test_class_method_resolves_across_modules(index) -> None:
    caller = make("src.train", "load_checkpoint")

    resolved = resolve_target("RecurrentLanguageModel.num_parameters", caller, index, set())

    assert resolved is not None
    assert resolved.module == "src.model"
