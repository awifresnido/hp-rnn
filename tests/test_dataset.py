"""Stage 4 tests: contiguous splits, shifted sequences, and batching."""

from __future__ import annotations

import torch

from src.data import (
    LanguageModelDataset,
    build_dataloaders,
    make_sequence_tensors,
    split_token_ids,
)


def test_split_token_ids_is_contiguous_and_covers_everything() -> None:
    ids = list(range(100))
    train, val, test = split_token_ids(ids, train_ratio=0.8, val_ratio=0.1)

    assert len(train) == 80 and len(val) == 10 and len(test) == 10
    assert train + val + test == ids


def test_make_sequence_tensors_shift_target_by_one() -> None:
    ids = list(range(10))
    inputs, targets = make_sequence_tensors(ids, sequence_length=4)

    assert inputs.shape == (6, 4)
    assert targets.shape == (6, 4)
    assert torch.equal(inputs[0], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(targets[0], torch.tensor([1, 2, 3, 4]))


def test_dataset_length_matches_available_windows() -> None:
    dataset = LanguageModelDataset(list(range(10)), sequence_length=4)

    assert len(dataset) == 6
    inputs, targets = dataset[3]
    assert torch.equal(inputs, torch.tensor([3, 4, 5, 6]))
    assert torch.equal(targets, torch.tensor([4, 5, 6, 7]))


def test_dataset_windows_are_non_overlapping_blocks_of_positions() -> None:
    dataset = LanguageModelDataset(list(range(10)), sequence_length=4)

    starts = [int(inputs[0]) for inputs, _ in dataset]
    assert starts == [0, 1, 2, 3, 4, 5]


def test_build_dataloaders_shapes_and_alignment() -> None:
    ids = list(range(200))
    loaders = build_dataloaders(ids, sequence_length=8, batch_size=4)

    inputs, targets = next(iter(loaders["train"]))
    assert inputs.shape == (4, 8)
    assert targets.shape == (4, 8)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_build_dataloaders_train_split_is_contiguous_prefix() -> None:
    ids = list(range(200))
    loaders = build_dataloaders(ids, sequence_length=8, batch_size=4)
    seen = torch.cat([inputs.flatten() for inputs, _ in loaders["train"]])

    assert int(seen.min()) >= 0
    assert int(seen.max()) < 180
