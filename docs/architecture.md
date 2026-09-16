# Repository architecture

Generated from the source by `scripts/depgraph.py` + `scripts/render_architecture.py`.
Regenerate after changing code:

```bash
python scripts/depgraph.py && python scripts/render_architecture.py
```

Edges are parsed from the real call sites; if a symbol named here disappears,
the renderer fails instead of drawing a stale diagram.

What is deliberately **not** resolved: calls made on an instance, such as
`vocab.encode(...)` or `model.eval()`. The qualifier is a runtime variable, so
matching its trailing name would invent edges — an earlier version linked a bare
`model.eval()` to a test helper that happened to be named `model`. Only bare
names, `self.method`, and `Class.method` resolve.

## Layers

```
L0  CLI entry points
     argparse mains; each one is a stage you can run
     compare.main, data.main, evaluate.main, generate.main, train.main, render_architecture.main

L1  Data preparation — Stages 2-4
     EPUB -> corpus -> tokens -> ids -> splits
     data.extract_epub_books, data.extract_epub_chapters, data.book_start_indices, data.clean_text, data.tokenize, Vocabulary, data.book_token_counts, data.split_token_ids_by_books, LanguageModelDataset, data.build_split_dataloaders, data.prepare_dataset

L2  Model — Stage 5
     one module, three architectures
     ModelConfig, RecurrentLanguageModel

L3  Train / evaluate / generate — Stages 6-9
     the training loop, perplexity, and decoding
     train.overfit_one_batch, train.clip_gradients_, train.resolve_device, train.load_processed, train.train_model, train.save_checkpoint, train.load_checkpoint, evaluate.sequence_cross_entropy, evaluate.perplexity_from_loss, evaluate.split_losses, evaluate.evaluate_checkpoint, generate.next_word_predictions, generate.sample_next_token, generate.generate_tokens, generate.generate, generate.join_tokens

L4  Compare — Stage 14
     tables derived from recorded runs
     RunSummary, compare.discover_runs, compare.load_run, compare.format_table

```

## Third-party libraries

- **bs4** — used by src.data
- **ebooklib** — used by src.data
- **pytest** — used by tests.conftest, tests.test_depgraph, tests.test_evaluate, tests.test_generation, tests.test_model, tests.test_split
- **torch** — used by src.data, src.evaluate, src.generate, src.model, src.train, tests.test_dataset, tests.test_evaluate, tests.test_generation, tests.test_model, tests.test_split, tests.test_training

## How the three architectures differ

Everything except one dictionary lookup is shared. `ModelConfig.cell`
selects the cell in `src/model.py`:

```text
Embedding -> _CELLS[cell] -> Dropout -> Linear -> [B, T, V] logits
                 |
                 +-- "rnn"  -> nn.RNN
                 +-- "gru"  -> nn.GRU
                 +-- "lstm" -> nn.LSTM
```

| Variant | cell | torch module | what changes |
| --- | --- | --- | --- |
| Vanilla RNN | `rnn` | `nn.RNN` | no gate: the hidden state is overwritten each step |
| GRU | `gru` | `nn.GRU` | update + reset gates decide what to keep |
| LSTM | `lstm` | `nn.LSTM` | input/forget/output gates plus a cell state |
| Improved LSTM | `lstm` | `nn.LSTM` | same cell, more capacity + dropout 0.4 |

## Modules

### `src/compare.py` — 199 lines

> Aggregate finished runs into the Stage 14 model comparison.

- internal: src.evaluate.evaluate_checkpoint, src.train.load_checkpoint
- third-party: none
- stdlib: __future__.annotations, argparse, collections.abc.Iterable, collections.abc.Sequence, dataclasses.dataclass, dataclasses.replace, json, math, pathlib.Path

Classes:
- `RunSummary` — One row of the comparison table.
  - methods: sort_key

Functions (with the repo symbols they call):
- `history_path(run_name, results_dir)` — 
  - calls: `Path`
- `discover_runs(results_dir)` — Find finished runs, ordered by architecture then name.
  - calls: `Path`, `glob`, `len`, `sorted`
- `load_run(run_name, results_dir, checkpoint_dir)` — Build one summary from a run's history JSON and best checkpoint.
  - calls: `FileNotFoundError`, `Path`, `RunSummary`, `checkpoint.exists`, `float`, `history_path`, `json.loads`, `len`, `load_checkpoint`, `math.exp`, `min`, `model.num_parameters`, `path.exists`, `path.read_text`, `payload.get`, `record.get`, `round`, `str`, `sum`, `train_config.get`
- `format_table(summaries)` — Render the comparison table; unevaluated test perplexity shows as pending.
  - calls: `join`, `lines.append`
- `to_markdown(summaries)` — Render the same results as a markdown table for the README/report.
  - calls: `join`, `lines.append`
- `_build_parser()` — 
  - calls: `Path`, `argparse.ArgumentParser`, `parser.add_argument`
- `main(argv)` — 
  - calls: `Path`, `_build_parser`, `args.markdown.parent.mkdir`, `args.markdown.write_text`, `discover_runs`, `evaluate_checkpoint`, `format_table`, `load_run`, `parse_args`, `print`, `replace`, `round`, `scored.append`, `sorted`, `to_markdown`

### `src/data.py` — 567 lines

> Corpus extraction, cleaning, tokenization, vocabulary, and sequence datasets.

- internal: none
- third-party: bs4.BeautifulSoup, bs4.XMLParsedAsHTMLWarning, ebooklib.ITEM_DOCUMENT, ebooklib.epub, torch, torch.utils.data.DataLoader, torch.utils.data.Dataset
- stdlib: __future__.annotations, argparse, collections.Counter, collections.abc.Iterable, collections.abc.Mapping, collections.abc.Sequence, dataclasses.dataclass, dataclasses.field, json, pathlib.Path, re, unicodedata, warnings

Classes:
- `CorpusStats` — Word-level statistics of a cleaned corpus.
- `Vocabulary` — A word-level vocabulary with reserved ``<PAD>`` and ``<UNK>`` entries.
  - methods: build, __len__, size, encode, decode, save, load
- `LanguageModelDataset(Dataset)` — Sliding windows where the target is the input shifted by one token.
  - methods: __init__, __len__, __getitem__
- `PreparedDataset` — Everything :func:`prepare_dataset` produced, ready to train on.

Functions (with the repo symbols they call):
- `tokenize(text)` — Split text into word and punctuation tokens, preserving case.
  - calls: `CHARACTER_MAP.items`, `TOKEN_PATTERN.findall`, `normalized.replace`
- `strip_boilerplate(text)` — Keep only the text between Project Gutenberg START/END markers, if present.
  - calls: `GUTENBERG_END.search`, `GUTENBERG_START.search`, `end.start`, `start.end`
- `clean_text(raw)` — Apply only the basic cleaning a word-level corpus needs.
  - calls: `ARTIFACT_LINE.match`, `CHARACTER_MAP.items`, `LINEBREAK_HYPHEN.sub`, `join`, `joined.strip`, `kept.append`, `re.sub`, `replace`, `strip`, `strip_boilerplate`, `text.replace`, `text.split`, `unicodedata.normalize`
- `extract_epub_chapters(epub_path)` — Return ``(heading, text)`` for every content document, in spine order.
  - calls: `BeautifulSoup`, `book.get_item_with_id`, `chapters.append`, `decode`, `epub.read_epub`, `found.get_text`, `item.get_content`, `item.get_type`, `soup`, `soup.find`, `soup.get_text`, `str`, `tag.decompose`, `text.strip`, `warnings.catch_warnings`, `warnings.filterwarnings`
- `book_start_indices(chapters)` — Indices of chapters that open a book, in reading order.
  - calls: `enumerate`, `heading.upper`, `title.upper`
- `extract_epub_books(epub_path)` — Return the cleaned text of each book in the omnibus, in reading order.
  - calls: `book_start_indices`, `clean_text`, `extract_epub_chapters`, `join`, `len`, `zip`
- `extract_epub_text(epub_path)` — Extract the readable text of an EPUB in spine (reading) order.
  - calls: `extract_epub_books`, `join`
- `corpus_statistics(text, top_n)` — Count characters, tokens, distinct tokens, and the most common tokens.
  - calls: `CorpusStats`, `Counter`, `counts.most_common`, `len`, `tokenize`
- `split_token_ids(ids, train_ratio, val_ratio)` — Split token IDs into contiguous train/validation/test sections.
  - calls: `int`, `len`, `list`
- `book_token_counts(books)` — Token count of each book, in reading order.
  - calls: `len`, `tokenize`
- `split_token_ids_by_books(ids, counts, train_books)` — Split token IDs by whole books: first ``train_books`` train, then val, then test.
  - calls: `ValueError`, `len`, `list`, `sum`
- `make_sequence_tensors(ids, sequence_length)` — Build shifted ``(input, target)`` tensors of shape ``[N, sequence_length]``.
  - calls: `ValueError`, `contiguous`, `data.unfold`, `len`, `list`, `torch.tensor`
- `build_split_dataloaders(splits, sequence_length, batch_size)` — Create one loader per named split, skipping splits too short to window.
  - calls: `DataLoader`, `LanguageModelDataset`, `len`, `splits.items`
- `build_dataloaders(ids, sequence_length, batch_size, train_ratio, val_ratio)` — Create train/validation/test loaders over one contiguous ratio split.
  - calls: `build_split_dataloaders`, `split_token_ids`
- `prepare_corpus(epub_path, output_dir)` — Extract and clean an EPUB into ``output_dir/corpus.txt``.
  - calls: `Path`, `corpus_path.write_text`, `extract_epub_text`, `output_dir.mkdir`
- `prepare_dataset(epub_path, output_dir, vocab_size, sequence_length, split_by, train_books, train_ratio, val_ratio)` — Full Stage 2-4 preparation: corpus, vocabulary, token IDs, and splits.
  - calls: `Path`, `PreparedDataset`, `ValueError`, `Vocabulary.build`, `book_token_counts`, `corpus_path.write_text`, `corpus_statistics`, `extract_epub_books`, `join`, `len`, `output_dir.mkdir`, `split_token_ids`, `split_token_ids_by_books`, `tokenize`, `torch.save`, `torch.tensor`, `vocab.encode`, `vocab.save`
- `format_statistics(stats, vocab_size, sequence_length)` — Render corpus statistics the way the project brief asks for them.
  - calls: `join`
- `_build_parser()` — 
  - calls: `Path`, `argparse.ArgumentParser`, `inspect.add_argument`, `parser.add_subparsers`, `prepare.add_argument`, `sub.add_parser`
- `main(argv)` — 
  - calls: `Vocabulary.load`, `_build_parser`, `bool`, `build_dataloaders`, `enumerate`, `format_statistics`, `int`, `iter`, `join`, `len`, `max`, `min`, `next`, `parse_args`, `prepare_dataset`, `print`, `range`, `tolist`, `torch.equal`, `torch.load`, `tuple`, `vocab.decode`

### `src/evaluate.py` — 153 lines

> Independent evaluation: split losses and perplexity from a saved checkpoint.

- internal: src.data.build_dataloaders, src.data.build_split_dataloaders, src.model.ModelConfig, src.train.load_checkpoint, src.train.load_processed, src.train.resolve_device
- third-party: torch, torch.nn, torch.utils.data.DataLoader
- stdlib: __future__.annotations, argparse, collections.abc.Mapping, collections.abc.Sequence, dataclasses.dataclass, math, pathlib.Path

Classes:
- `EvaluationReport` — Loss, perplexity, and model size for one checkpoint.
  - methods: table

Functions (with the repo symbols they call):
- `sequence_cross_entropy(logits, targets)` — Cross entropy over every predicted position in a ``[B, T, V]`` batch.
  - calls: `logits.reshape`, `logits.size`, `nn.functional.cross_entropy`, `targets.reshape`
- `perplexity_from_loss(loss)` — Convert a cross-entropy loss in nats into perplexity.
  - calls: `math.exp`
- `split_losses(model, loaders, device)` — Average cross-entropy loss over each split, with the model in eval mode.
  - calls: `ValueError`, `eval`, `float`, `inputs.to`, `loaders.items`, `model`, `model.to`, `sequence_cross_entropy`, `targets.numel`, `targets.to`, `torch.device`, `torch.no_grad`
- `evaluate_checkpoint(checkpoint_path, processed_dir, device, batch_size, sequence_length)` — Evaluate a saved checkpoint against the prepared corpus splits.
  - calls: `EvaluationReport`, `Path`, `build_dataloaders`, `build_split_dataloaders`, `contents.model.num_parameters`, `load_checkpoint`, `load_processed`, `losses.items`, `perplexity_from_loss`, `resolve_device`, `split_losses`
- `_build_parser()` — 
  - calls: `Path`, `argparse.ArgumentParser`, `parser.add_argument`
- `main(argv)` — 
  - calls: `_build_parser`, `evaluate_checkpoint`, `parse_args`, `print`, `report.table`

### `src/generate.py` — 266 lines

> Next-word prediction and recursive text generation.

- internal: src.data.Vocabulary, src.data.tokenize, src.model.RecurrentLanguageModel, src.train.load_checkpoint, src.train.resolve_device
- third-party: torch
- stdlib: __future__.annotations, argparse, collections.abc.Sequence, dataclasses.dataclass, pathlib.Path

Classes:
- `Prediction` — One candidate next word and its probability.

Functions (with the repo symbols they call):
- `tokenize_prompt(vocab, prompt)` — Tokenize a prompt with the corpus tokenizer and encode it to IDs.
  - calls: `ValueError`, `tokenize`, `vocab.encode`
- `check_vocab_matches_model(vocab, model)` — Fail clearly when a checkpoint's vocabulary size disagrees with the corpus.
  - calls: `ValueError`
- `next_word_predictions(model, vocab, prompt)` — Return the ``top_n`` most probable next words after ``prompt``.
  - calls: `Prediction`, `check_vocab_matches_model`, `eval`, `float`, `indices.tolist`, `int`, `min`, `model`, `model.to`, `tokenize_prompt`, `torch.no_grad`, `torch.softmax`, `torch.tensor`, `torch.topk`, `values.tolist`, `zip`
- `sample_next_token(logits)` — Choose the next token from ``logits`` of shape ``[1, V]``.
  - calls: `logits.argmax`, `scaled.size`, `squeeze`, `torch.multinomial`, `torch.softmax`, `torch.topk`, `torch.where`
- `generate_tokens(model, vocab, prompt)` — Generate ``words`` new words after ``prompt`` and return them as tokens.
  - calls: `ValueError`, `check_vocab_matches_model`, `eval`, `generated.append`, `ids.append`, `int`, `item`, `manual_seed`, `model`, `model.to`, `range`, `sample_next_token`, `to`, `tokenize_prompt`, `torch.Generator`, `torch.no_grad`, `torch.tensor`, `unsqueeze`
- `generate(model, vocab, prompt)` — Generate ``words`` new words after ``prompt`` and render them as text.
  - calls: `generate_tokens`, `join_tokens`
- `join_tokens(prompt, generated)` — Reattach generated words to the prompt without disturbing the prompt text.
  - calls: `prompt.rstrip`, `token.startswith`
- `format_predictions(predictions)` — Render predictions as a token/probability table.
  - calls: `join`, `len`, `max`
- `build_parser()` — 
  - calls: `Path`, `argparse.ArgumentParser`, `parser.add_argument`
- `main(argv)` — 
  - calls: `Path`, `Vocabulary.load`, `build_parser`, `format_predictions`, `generate`, `len`, `load_checkpoint`, `next_word_predictions`, `parse_args`, `print`, `resolve_device`

### `src/model.py` — 88 lines

> The recurrent language model: embedding -> recurrent cell -> vocabulary logits.

- internal: none
- third-party: torch, torch.nn
- stdlib: __future__.annotations, dataclasses.asdict, dataclasses.dataclass

Classes:
- `ModelConfig` — Architecture settings for :class:`RecurrentLanguageModel`.
  - methods: __post_init__, to_dict, from_dict
- `RecurrentLanguageModel(nn.Module)` — Word-level language model: ``[B, T]`` token IDs -> ``[B, T, V]`` logits.
  - methods: __init__, forward, num_parameters

### `src/train.py` — 435 lines

> Training: the one-batch diagnostic, gradient clipping, and the training loop.

- internal: src.data.build_dataloaders, src.data.build_split_dataloaders, src.evaluate.sequence_cross_entropy, src.evaluate.split_losses, src.model.CELL_TYPES, src.model.ModelConfig, src.model.RecurrentLanguageModel
- third-party: torch, torch.nn
- stdlib: __future__.annotations, argparse, collections.abc.Callable, collections.abc.Sequence, dataclasses.asdict, dataclasses.dataclass, dataclasses.field, json, pathlib.Path, random, time

Classes:
- `TrainConfig` — Everything that affects a training run, so runs stay comparable.
  - methods: model_config, to_dict
- `EpochRecord` — One row of the training history.
- `TrainResult` — Outcome of :func:`train_model`.
- `CheckpointContents` — A checkpoint loaded back into a usable model plus its metadata.
- `ProcessedData` — Prepared corpus as training consumes it.

Functions (with the repo symbols they call):
- `resolve_device(requested)` — Turn a device request into a device, refusing a silent CPU fallback.
  - calls: `RuntimeError`, `torch.cuda.is_available`, `torch.device`
- `clip_gradients_(model, max_norm)` — Clip gradients in place and return the total norm measured beforehand.
  - calls: `ValueError`, `float`, `model.parameters`, `nn.utils.clip_grad_norm_`
- `set_seed(seed)` — Seed Python and torch so a run can be repeated.
  - calls: `random.seed`, `torch.cuda.is_available`, `torch.cuda.manual_seed_all`, `torch.manual_seed`
- `save_checkpoint(path)` — Write a self-describing checkpoint (architecture + progress in one file).
  - calls: `Path`, `model.state_dict`, `model_config.to_dict`, `optimizer.state_dict`, `path.parent.mkdir`, `torch.save`
- `load_checkpoint(path, device)` — Rebuild a model from a checkpoint, evaluating on ``device``.
  - calls: `CheckpointContents`, `ModelConfig.from_dict`, `Path`, `RecurrentLanguageModel`, `eval`, `float`, `int`, `model.load_state_dict`, `model.to`, `payload.get`, `str`, `torch.device`, `torch.load`
- `overfit_one_batch(model, inputs, targets)` — Train repeatedly on one batch and return the loss after each step.
  - calls: `clip_gradients_`, `history.append`, `inputs.to`, `item`, `loss.backward`, `loss.detach`, `model`, `model.parameters`, `model.to`, `optimizer.step`, `optimizer.zero_grad`, `range`, `sequence_cross_entropy`, `set_seed`, `targets.to`, `torch.device`, `torch.optim.AdamW`, `train`
- `load_processed(processed_dir)` — Read token IDs, sequence length, vocabulary size, and any explicit splits.
  - calls: `Path`, `ProcessedData`, `int`, `json.loads`, `len`, `list`, `payload.get`, `raw_splits.items`, `read_text`, `str`, `tolist`, `torch.load`
- `train_model(config)` — Run the one-batch check (optionally) and train, keeping the best epoch.
  - calls: `EpochRecord`, `Path`, `RecurrentLanguageModel`, `TrainResult`, `build_dataloaders`, `build_split_dataloaders`, `clip_gradients_`, `config.model_config`, `inputs.to`, `item`, `iter`, `load_processed`, `log`, `loss.backward`, `loss.detach`, `model`, `model.parameters`, `model.train`, `next`, `optimizer.step`, `optimizer.zero_grad`, `overfit_one_batch`, `range`, `resolve_device`, `result.history.append`, `save_checkpoint`, `sequence_cross_entropy`, `set_seed`, `split_losses`, `str`, `targets.numel`, `targets.to`, `time.perf_counter`, `to`, `torch.optim.AdamW`
- `format_history(result)` — Render the per-epoch table the training loop prints.
  - calls: `join`, `lines.append`
- `_build_parser()` — 
  - calls: `Path`, `argparse.ArgumentParser`, `parser.add_argument`
- `main(argv)` — 
  - calls: `Path`, `TrainConfig`, `_build_parser`, `asdict`, `config.to_dict`, `format_history`, `history_path.parent.mkdir`, `history_path.write_text`, `json.dumps`, `parse_args`, `print`, `resolve_device`, `str`, `train_model`

### `tests/conftest.py` — 148 lines

> Shared fixtures: a tiny synthetic EPUB so tests never touch real book text.

- internal: none
- third-party: pytest
- stdlib: __future__.annotations, pathlib.Path, zipfile

Functions (with the repo symbols they call):
- `write_epub(path)` — Write a minimal two-chapter EPUB in spine order.
  - calls: `z.writestr`, `zipfile.ZipFile`
- `fixture_epub(tmp_path)` — 
  - calls: `write_epub`
- `_chapter(heading, body, title)` — 
- `write_two_book_epub(path)` — Write an omnibus EPUB with two detectable books of two chapters each.
  - calls: `_chapter`, `z.writestr`, `zipfile.ZipFile`
- `two_book_epub(tmp_path)` — 
  - calls: `write_two_book_epub`
- `fixture_tokens()` — 

### `tests/test_compare.py` — 147 lines

> Stage 14 tests: aggregating runs into the model comparison table.

- internal: src.compare.RunSummary, src.compare.format_table, src.compare.load_run, src.model.ModelConfig, src.model.RecurrentLanguageModel, src.train.save_checkpoint
- third-party: none
- stdlib: __future__.annotations, json, math

Functions (with the repo symbols they call):
- `write_run(results_dir, checkpoint_dir, run_name, cell)` — Write the history JSON and best checkpoint that a finished run leaves behind.
  - calls: `ModelConfig`, `RecurrentLanguageModel`, `checkpoint_dir.mkdir`, `json.dumps`, `range`, `results_dir.mkdir`, `save_checkpoint`, `str`, `write_text`
- `test_load_run_reads_metrics_from_history_and_checkpoint(tmp_path)` — 
  - calls: `isinstance`, `load_run`, `write_run`
- `test_val_perplexity_is_exp_of_best_val_loss(tmp_path)` — 
  - calls: `load_run`, `math.exp`, `math.log`, `round`, `write_run`
- `test_table_lists_every_run_in_the_requested_order(tmp_path)` — 
  - calls: `format_table`, `load_run`, `table.index`, `write_run`
- `test_unevaluated_test_perplexity_is_shown_as_pending(tmp_path)` — 
  - calls: `format_table`, `load_run`, `write_run`
- `test_missing_metrics_render_as_not_available(tmp_path)` — 
  - calls: `RunSummary`, `format_table`
- `test_empty_run_list_produces_a_header_only_table()` — 
  - calls: `format_table`, `len`, `table.splitlines`
- `test_missing_history_file_is_reported_clearly(tmp_path)` — 
  - calls: `AssertionError`, `load_run`, `str`

### `tests/test_corpus.py` — 63 lines

> Stage 2 tests: EPUB extraction, basic cleaning, and corpus statistics.

- internal: src.data.clean_text, src.data.corpus_statistics, src.data.extract_epub_text, src.data.prepare_corpus
- third-party: none
- stdlib: __future__.annotations, pathlib.Path

Functions (with the repo symbols they call):
- `test_extract_epub_text_follows_spine_order(fixture_epub)` — 
  - calls: `extract_epub_text`, `text.index`
- `test_extract_epub_text_drops_markup_and_navigation(fixture_epub)` — 
  - calls: `extract_epub_text`
- `test_clean_text_normalizes_typographic_quotes()` — 
  - calls: `clean_text`
- `test_clean_text_joins_line_break_hyphenation()` — 
  - calls: `clean_text`
- `test_clean_text_drops_page_numbers_and_artifact_lines()` — 
  - calls: `clean_text`
- `test_clean_text_collapses_excess_whitespace()` — 
  - calls: `clean_text`
- `test_corpus_statistics_counts_characters_words_and_uniques()` — 
  - calls: `corpus_statistics`
- `test_prepare_corpus_writes_processed_files(fixture_epub, tmp_path)` — 
  - calls: `corpus_path.read_text`, `prepare_corpus`

### `tests/test_dataset.py` — 65 lines

> Stage 4 tests: contiguous splits, shifted sequences, and batching.

- internal: src.data.LanguageModelDataset, src.data.build_dataloaders, src.data.make_sequence_tensors, src.data.split_token_ids
- third-party: torch
- stdlib: __future__.annotations

Functions (with the repo symbols they call):
- `test_split_token_ids_is_contiguous_and_covers_everything()` — 
  - calls: `len`, `list`, `range`, `split_token_ids`
- `test_make_sequence_tensors_shift_target_by_one()` — 
  - calls: `list`, `make_sequence_tensors`, `range`, `torch.equal`, `torch.tensor`
- `test_dataset_length_matches_available_windows()` — 
  - calls: `LanguageModelDataset`, `len`, `list`, `range`, `torch.equal`, `torch.tensor`
- `test_dataset_windows_are_non_overlapping_blocks_of_positions()` — 
  - calls: `LanguageModelDataset`, `int`, `list`, `range`
- `test_build_dataloaders_shapes_and_alignment()` — 
  - calls: `build_dataloaders`, `iter`, `list`, `next`, `range`, `torch.equal`
- `test_build_dataloaders_train_split_is_contiguous_prefix()` — 
  - calls: `build_dataloaders`, `inputs.flatten`, `int`, `list`, `range`, `seen.max`, `seen.min`, `torch.cat`

### `tests/test_depgraph.py` — 99 lines

> Tests for the dependency-graph resolver.

- internal: none
- third-party: depgraph.Function, depgraph.resolve_target, pytest
- stdlib: __future__.annotations, pathlib.Path, sys

Functions (with the repo symbols they call):
- `make(module, name, owner)` — 
  - calls: `Function`
- `index()` — 
  - calls: `append`, `index.setdefault`, `make`
- `test_calls_on_lowercase_variables_never_resolve(index)` — 
  - calls: `make`, `resolve_target`, `set`
- `test_bare_name_resolves_within_the_same_module(index)` — 
  - calls: `make`, `resolve_target`, `set`
- `test_bare_name_resolves_through_an_internal_import(index)` — 
  - calls: `make`, `resolve_target`
- `test_bare_name_does_not_cross_into_an_unimported_module(index)` — 
  - calls: `make`, `resolve_target`, `set`
- `test_self_method_resolves_to_the_owning_class(index)` — 
  - calls: `make`, `resolve_target`, `set`
- `test_class_method_resolves_across_modules(index)` — 
  - calls: `make`, `resolve_target`, `set`

### `tests/test_evaluate.py` — 87 lines

> Stage 7 tests: independent evaluation, loss, and perplexity.

- internal: src.data.build_dataloaders, src.evaluate.evaluate_checkpoint, src.evaluate.perplexity_from_loss, src.evaluate.sequence_cross_entropy, src.evaluate.split_losses, src.model.ModelConfig, src.model.RecurrentLanguageModel, src.train.save_checkpoint
- third-party: pytest, torch
- stdlib: __future__.annotations, math

Functions (with the repo symbols they call):
- `test_perplexity_is_the_exponential_of_cross_entropy()` — 
  - calls: `math.log`, `perplexity_from_loss`, `pytest.approx`
- `test_sequence_cross_entropy_matches_manual_flattened_computation()` — 
  - calls: `logits.reshape`, `manual_seed`, `sequence_cross_entropy`, `targets.reshape`, `torch.Generator`, `torch.allclose`, `torch.nn.functional.cross_entropy`, `torch.randint`, `torch.randn`
- `test_split_losses_reports_a_loss_for_every_split()` — 
  - calls: `ModelConfig`, `RecurrentLanguageModel`, `all`, `build_dataloaders`, `losses.values`, `manual_seed`, `math.isfinite`, `set`, `split_losses`, `tolist`, `torch.Generator`, `torch.randint`
- `test_evaluate_checkpoint_works_without_any_training(tmp_path)` — 
  - calls: `ModelConfig`, `RecurrentLanguageModel`, `evaluate_checkpoint`, `manual_seed`, `math.exp`, `model.num_parameters`, `processed_dir.mkdir`, `pytest.approx`, `save_checkpoint`, `set`, `torch.Generator`, `torch.randint`, `torch.save`

### `tests/test_generation.py` — 165 lines

> Stages 8-9 tests: next-word probabilities, greedy, temperature, and top-k.

- internal: src.data.Vocabulary, src.generate.Prediction, src.generate.generate, src.generate.generate_tokens, src.generate.next_word_predictions, src.generate.sample_next_token, src.generate.tokenize_prompt, src.model.ModelConfig, src.model.RecurrentLanguageModel
- third-party: pytest, torch
- stdlib: __future__.annotations, math

Functions (with the repo symbols they call):
- `vocab()` — 
  - calls: `Vocabulary.build`
- `model(vocab)` — A model whose output layer matches the fixture vocabulary exactly.
  - calls: `ModelConfig`, `RecurrentLanguageModel`, `eval`, `torch.manual_seed`
- `test_tokenize_prompt_reuses_the_corpus_tokenizer(vocab)` — 
  - calls: `tokenize_prompt`, `vocab.encode`
- `test_predictions_are_ranked_and_sum_to_at_most_one(model, vocab)` — 
  - calls: `all`, `isinstance`, `len`, `next_word_predictions`, `sorted`, `sum`
- `test_top_n_is_capped_by_vocabulary_size(model, vocab)` — 
  - calls: `len`, `next_word_predictions`
- `test_greedy_selection_picks_the_highest_logit()` — 
  - calls: `chosen.item`, `int`, `sample_next_token`, `torch.tensor`
- `test_top_k_of_one_equals_greedy()` — 
  - calls: `greedy.item`, `int`, `item`, `logits.argmax`, `manual_seed`, `sample_next_token`, `top_one.item`, `torch.Generator`, `torch.randn`
- `test_zero_temperature_falls_back_to_greedy()` — 
  - calls: `chosen.item`, `int`, `item`, `logits.argmax`, `manual_seed`, `sample_next_token`, `torch.Generator`, `torch.randn`
- `test_generation_is_reproducible_with_a_seed(model, vocab)` — 
  - calls: `generate`
- `test_generation_produces_the_requested_number_of_words(model, vocab)` — 
  - calls: `all`, `generate_tokens`, `len`
- `test_rendered_generation_keeps_the_prompt_intact(model, vocab)` — 
  - calls: `generate`, `len`, `text.startswith`
- `test_only_the_last_context_tokens_influence_generation(model, vocab)` — 
  - calls: `generate_tokens`, `join`
- `test_temperature_affects_the_sampling_distribution(model, vocab)` — 
  - calls: `generate`
- `test_mismatched_vocabulary_and_model_are_rejected(model)` — 
  - calls: `Vocabulary.build`, `next_word_predictions`, `pytest.raises`
- `test_prediction_probabilities_are_finite(model, vocab)` — 
  - calls: `all`, `math.isfinite`, `next_word_predictions`

### `tests/test_model.py` — 95 lines

> Stage 5 tests: recurrent language model shapes, gradients, and configuration.

- internal: src.model.CELL_TYPES, src.model.ModelConfig, src.model.RecurrentLanguageModel
- third-party: pytest, torch
- stdlib: __future__.annotations

Functions (with the repo symbols they call):
- `build_model(cell)` — 
  - calls: `ModelConfig`, `RecurrentLanguageModel`, `settings.update`
- `sample_inputs()` — 
  - calls: `torch.randint`
- `test_model_maps_batch_and_time_to_vocabulary_logits(cell)` — 
  - calls: `build_model`, `pytest.mark.parametrize`, `sample_inputs`
- `test_backward_produces_gradients_for_every_parameter(cell)` — 
  - calls: `build_model`, `item`, `logits.reshape`, `loss.backward`, `model`, `model.embedding.weight.grad.abs`, `model.head.weight.grad.abs`, `model.named_parameters`, `pytest.mark.parametrize`, `sample_inputs`, `sum`, `targets.reshape`, `torch.nn.functional.cross_entropy`, `torch.randint`
- `test_parameter_count_is_positive(cell)` — 
  - calls: `build_model`, `num_parameters`, `pytest.mark.parametrize`
- `test_parameter_count_grows_with_hidden_size()` — 
  - calls: `build_model`, `num_parameters`
- `test_dropout_keeps_output_shape_valid()` — 
  - calls: `all`, `build_model`, `sample_inputs`, `torch.isfinite`
- `test_eval_mode_is_deterministic()` — 
  - calls: `build_model`, `eval`, `model`, `sample_inputs`, `torch.equal`, `torch.no_grad`
- `test_config_round_trips_through_dict()` — 
  - calls: `ModelConfig`, `ModelConfig.from_dict`, `config.to_dict`
- `test_unknown_cell_type_is_rejected()` — 
  - calls: `build_model`, `pytest.raises`

### `tests/test_split.py` — 107 lines

> Book-level splitting: split on whole books rather than a ratio of tokens.

- internal: src.data.book_token_counts, src.data.build_split_dataloaders, src.data.extract_epub_books, src.data.extract_epub_text, src.data.prepare_dataset, src.data.split_token_ids_by_books, src.data.tokenize
- third-party: pytest, torch
- stdlib: __future__.annotations, pathlib.Path

Functions (with the repo symbols they call):
- `test_extract_epub_books_groups_chapters_by_book_opening(two_book_epub)` — 
  - calls: `extract_epub_books`, `len`
- `test_book_token_counts_sum_to_the_whole_corpus(two_book_epub)` — 
  - calls: `book_token_counts`, `extract_epub_books`, `extract_epub_text`, `len`, `sum`, `tokenize`
- `test_split_token_ids_by_books_takes_whole_books_contiguously()` — 
  - calls: `len`, `list`, `range`, `split_token_ids_by_books`
- `test_split_token_ids_by_books_rejects_a_count_mismatch()` — 
  - calls: `list`, `pytest.raises`, `range`, `split_token_ids_by_books`
- `test_split_token_ids_by_books_rejects_impossible_book_counts()` — 
  - calls: `list`, `pytest.raises`, `range`, `split_token_ids_by_books`
- `test_book_split_keeps_books_out_of_each_others_splits(two_book_epub)` — 
  - calls: `book_token_counts`, `extract_epub_books`, `len`, `list`, `range`, `split_token_ids_by_books`, `sum`
- `test_build_split_dataloaders_honours_explicit_splits()` — 
  - calls: `build_split_dataloaders`, `inputs.flatten`, `int`, `list`, `range`, `torch.cat`, `train_seen.max`, `val_seen.max`, `val_seen.min`
- `test_prepare_dataset_book_mode_records_the_splits(two_book_epub, tmp_path)` — 
  - calls: `len`, `prepare_dataset`, `set`, `torch.load`
- `test_prepare_dataset_ratio_mode_remains_the_default(fixture_epub, tmp_path)` — 
  - calls: `prepare_dataset`, `set`, `torch.load`

### `tests/test_tokenizer.py` — 76 lines

> Stage 3 tests: word-level tokenization, vocabulary, and ID round-trip.

- internal: src.data.Vocabulary, src.data.tokenize
- third-party: none
- stdlib: __future__.annotations

Functions (with the repo symbols they call):
- `test_tokenize_separates_words_and_punctuation()` — 
  - calls: `tokenize`
- `test_tokenize_keeps_contractions_and_capitalization()` — 
  - calls: `tokenize`
- `test_tokenize_handles_multicharacter_punctuation()` — 
  - calls: `tokenize`
- `test_vocabulary_reserves_pad_and_unk_ids()` — 
  - calls: `Vocabulary.build`
- `test_vocabulary_orders_by_frequency_then_first_appearance()` — 
  - calls: `Vocabulary.build`
- `test_vocabulary_respects_max_size_and_maps_rare_words_to_unk()` — 
  - calls: `Vocabulary.build`, `vocab.encode`
- `test_encode_decode_round_trip()` — 
  - calls: `Vocabulary.build`, `vocab.decode`, `vocab.encode`
- `test_vocabulary_save_and_load_round_trip(tmp_path)` — 
  - calls: `Vocabulary.build`, `Vocabulary.load`, `loaded.encode`, `vocab.encode`, `vocab.save`

### `tests/test_training.py` — 213 lines

> Stage 6 tests: one-batch overfit, gradient clipping, and checkpoint round-trip.

- internal: src.model.ModelConfig, src.model.RecurrentLanguageModel, src.train.TrainConfig, src.train.clip_gradients_, src.train.load_checkpoint, src.train.overfit_one_batch, src.train.save_checkpoint, src.train.train_model
- third-party: torch
- stdlib: __future__.annotations

Functions (with the repo symbols they call):
- `tiny_model(cell)` — 
  - calls: `ModelConfig`, `RecurrentLanguageModel`
- `tiny_batch(size)` — 
  - calls: `manual_seed`, `torch.Generator`, `torch.randint`
- `test_overfit_one_batch_reduces_loss_substantially()` — 
  - calls: `overfit_one_batch`, `tiny_batch`, `tiny_model`
- `test_overfit_one_batch_is_reproducible_with_a_seed()` — 
  - calls: `overfit_one_batch`, `tiny_batch`, `tiny_model`, `torch.manual_seed`
- `test_clip_gradients_bounds_the_total_norm()` — 
  - calls: `clip_gradients_`, `logits.reshape`, `loss.backward`, `model`, `model.parameters`, `p.grad.detach`, `sum`, `targets.reshape`, `tiny_batch`, `tiny_model`, `torch.nn.functional.cross_entropy`, `torch.sqrt`, `total.item`
- `test_checkpoint_round_trip_reproduces_predictions(tmp_path)` — 
  - calls: `eval`, `load_checkpoint`, `loaded.model`, `model`, `save_checkpoint`, `tiny_batch`, `tiny_model`, `torch.allclose`, `torch.manual_seed`, `torch.no_grad`
- `test_checkpoint_records_metadata_for_standalone_evaluation(tmp_path)` — 
  - calls: `load_checkpoint`, `save_checkpoint`, `tiny_model`
- `write_processed(dir_path, ids)` — Write the processed-data contract that training reads.
  - calls: `torch.save`
- `small_config(tmp_path, run_name, epochs)` — 
  - calls: `TrainConfig`
- `test_train_model_reports_each_epoch_as_it_finishes(tmp_path)` — 
  - calls: `manual_seed`, `small_config`, `torch.Generator`, `torch.randint`, `train_model`, `write_processed`
- `test_training_continues_when_no_log_callback_is_given(tmp_path)` — 
  - calls: `len`, `manual_seed`, `small_config`, `torch.Generator`, `torch.randint`, `train_model`, `write_processed`
- `test_train_model_saves_best_checkpoint_and_tracks_history(tmp_path)` — 
  - calls: `TrainConfig`, `exists`, `len`, `manual_seed`, `min`, `torch.Generator`, `torch.randint`, `torch.save`, `train_model`

### `scripts/check_diagram.py` — 54 lines

> Sanity-check the generated dependency-graph page.

- internal: none
- third-party: none
- stdlib: collections.Counter, html.parser.HTMLParser, pathlib.Path, re

Classes:
- `Checker(HTMLParser)` — no docstring
  - methods: __init__, handle_starttag, handle_endtag

### `scripts/render_architecture.py` — 568 lines

> Render docs/architecture.md and docs/dependency-graph.html from the parsed graph.

- internal: none
- third-party: none
- stdlib: __future__.annotations, html, json, pathlib.Path

Functions (with the repo symbols they call):
- `load_graph()` — 
  - calls: `GRAPH_PATH.exists`, `GRAPH_PATH.read_text`, `SystemExit`, `json.loads`
- `symbol_index(graph)` — Map every known symbol name to the modules that define it.
  - calls: `append`, `index.setdefault`
- `verify_layers(graph)` — Fail loudly if a layer names a symbol that no longer exists.
  - calls: `SystemExit`, `join`, `sorted`, `symbol_index`
- `module_of(graph, symbol)` — 
  - calls: `index.get`, `symbol_index`
- `layer_nodes(graph)` — Per layer, the (node_id, label) pairs actually defined in the code.
  - calls: `built.append`, `dict.fromkeys`, `index.get`, `len`, `module.split`, `nodes.append`, `symbol_index`
- `graph_edges(graph, nodes)` — Call edges whose two ends are both drawn, deduplicated.
  - calls: `drawn.append`, `enumerate`, `seen.add`, `set`
- `library_users(graph)` — Which modules import each third-party library.
  - calls: `append`, `imported.split`, `lib.split`, `set`, `sorted`, `users.items`, `users.setdefault`
- `render_markdown(graph)` — 
  - calls: `enumerate`, `join`, `layer_nodes`, `library_users`, `lines.append`, `sorted`, `split`
- `_wrap(text, width)` — 
  - calls: `len`, `lines.append`, `strip`, `text.split`
- `_node_width(label)` — 
  - calls: `int`, `len`, `max`, `min`
- `render_svg(graph, nodes, edges)` — Layered top-down graph: boxes per layer, curved edges behind them.
  - calls: `_node_width`, `_wrap`, `band_starts.append`, `enumerate`, `html.escape`, `join`, `len`, `lib.split`, `libraries.append`, `library.split`, `library_users`, `m.split`, `max`, `parts.append`
- `render_html(graph, svg)` — Assemble the standalone page: diagram, per-model panels, module cards.
  - calls: `html.escape`, `join`, `len`, `library_users`, `m.split`, `model_panels.append`, `module_cards.append`, `sorted`, `startswith`, `users.items`
- `main()` — 
  - calls: `DOCS.mkdir`, `SystemExit`, `graph_edges`, `html_path.relative_to`, `html_path.write_text`, `json.dumps`, `layer_nodes`, `len`, `load_graph`, `markdown_path.relative_to`, `markdown_path.write_text`, `print`, `read_text`, `render_html`, `render_markdown`, `render_svg`, `sum`, `verify_layers`, `write_text`

## Cross-module call edges

- `src.compare:load_run` → `src.train:load_checkpoint`
- `src.compare:main` → `src.evaluate:evaluate_checkpoint`
- `src.evaluate:evaluate_checkpoint` → `src.data:build_dataloaders`
- `src.evaluate:evaluate_checkpoint` → `src.data:build_split_dataloaders`
- `src.evaluate:evaluate_checkpoint` → `src.train:load_checkpoint`
- `src.evaluate:evaluate_checkpoint` → `src.train:load_processed`
- `src.evaluate:evaluate_checkpoint` → `src.train:resolve_device`
- `src.generate:main` → `src.data:Vocabulary.load`
- `src.generate:main` → `src.train:load_checkpoint`
- `src.generate:main` → `src.train:resolve_device`
- `src.generate:tokenize_prompt` → `src.data:tokenize`
- `src.train:load_checkpoint` → `src.model:ModelConfig.from_dict`
- `src.train:overfit_one_batch` → `src.evaluate:sequence_cross_entropy`
- `src.train:train_model` → `src.data:build_dataloaders`
- `src.train:train_model` → `src.data:build_split_dataloaders`
- `src.train:train_model` → `src.evaluate:sequence_cross_entropy`
- `src.train:train_model` → `src.evaluate:split_losses`
- `tests.test_compare:test_empty_run_list_produces_a_header_only_table` → `src.compare:format_table`
- `tests.test_compare:test_load_run_reads_metrics_from_history_and_checkpoint` → `src.compare:load_run`
- `tests.test_compare:test_missing_history_file_is_reported_clearly` → `src.compare:load_run`
- `tests.test_compare:test_missing_metrics_render_as_not_available` → `src.compare:format_table`
- `tests.test_compare:test_table_lists_every_run_in_the_requested_order` → `src.compare:format_table`
- `tests.test_compare:test_table_lists_every_run_in_the_requested_order` → `src.compare:load_run`
- `tests.test_compare:test_unevaluated_test_perplexity_is_shown_as_pending` → `src.compare:format_table`
- `tests.test_compare:test_unevaluated_test_perplexity_is_shown_as_pending` → `src.compare:load_run`
- `tests.test_compare:test_val_perplexity_is_exp_of_best_val_loss` → `src.compare:load_run`
- `tests.test_compare:write_run` → `src.train:save_checkpoint`
- `tests.test_corpus:test_clean_text_collapses_excess_whitespace` → `src.data:clean_text`
- `tests.test_corpus:test_clean_text_drops_page_numbers_and_artifact_lines` → `src.data:clean_text`
- `tests.test_corpus:test_clean_text_joins_line_break_hyphenation` → `src.data:clean_text`
- `tests.test_corpus:test_clean_text_normalizes_typographic_quotes` → `src.data:clean_text`
- `tests.test_corpus:test_corpus_statistics_counts_characters_words_and_uniques` → `src.data:corpus_statistics`
- `tests.test_corpus:test_extract_epub_text_drops_markup_and_navigation` → `src.data:extract_epub_text`
- `tests.test_corpus:test_extract_epub_text_follows_spine_order` → `src.data:extract_epub_text`
- `tests.test_corpus:test_prepare_corpus_writes_processed_files` → `src.data:prepare_corpus`
- `tests.test_dataset:test_build_dataloaders_shapes_and_alignment` → `src.data:build_dataloaders`
- `tests.test_dataset:test_build_dataloaders_train_split_is_contiguous_prefix` → `src.data:build_dataloaders`
- `tests.test_dataset:test_make_sequence_tensors_shift_target_by_one` → `src.data:make_sequence_tensors`
- `tests.test_dataset:test_split_token_ids_is_contiguous_and_covers_everything` → `src.data:split_token_ids`
- `tests.test_evaluate:test_evaluate_checkpoint_works_without_any_training` → `src.evaluate:evaluate_checkpoint`
- `tests.test_evaluate:test_evaluate_checkpoint_works_without_any_training` → `src.train:save_checkpoint`
- `tests.test_evaluate:test_perplexity_is_the_exponential_of_cross_entropy` → `src.evaluate:perplexity_from_loss`
- `tests.test_evaluate:test_sequence_cross_entropy_matches_manual_flattened_computation` → `src.evaluate:sequence_cross_entropy`
- `tests.test_evaluate:test_split_losses_reports_a_loss_for_every_split` → `src.data:build_dataloaders`
- `tests.test_evaluate:test_split_losses_reports_a_loss_for_every_split` → `src.evaluate:split_losses`
- `tests.test_generation:test_generation_is_reproducible_with_a_seed` → `src.generate:generate`
- `tests.test_generation:test_generation_produces_the_requested_number_of_words` → `src.generate:generate_tokens`
- `tests.test_generation:test_greedy_selection_picks_the_highest_logit` → `src.generate:sample_next_token`
- `tests.test_generation:test_mismatched_vocabulary_and_model_are_rejected` → `src.data:Vocabulary.build`
- `tests.test_generation:test_mismatched_vocabulary_and_model_are_rejected` → `src.generate:next_word_predictions`
- `tests.test_generation:test_only_the_last_context_tokens_influence_generation` → `src.generate:generate_tokens`
- `tests.test_generation:test_prediction_probabilities_are_finite` → `src.generate:next_word_predictions`
- `tests.test_generation:test_predictions_are_ranked_and_sum_to_at_most_one` → `src.generate:next_word_predictions`
- `tests.test_generation:test_rendered_generation_keeps_the_prompt_intact` → `src.generate:generate`
- `tests.test_generation:test_temperature_affects_the_sampling_distribution` → `src.generate:generate`
- `tests.test_generation:test_tokenize_prompt_reuses_the_corpus_tokenizer` → `src.generate:tokenize_prompt`
- `tests.test_generation:test_top_k_of_one_equals_greedy` → `src.generate:sample_next_token`
- `tests.test_generation:test_top_n_is_capped_by_vocabulary_size` → `src.generate:next_word_predictions`
- `tests.test_generation:test_zero_temperature_falls_back_to_greedy` → `src.generate:sample_next_token`
- `tests.test_generation:vocab` → `src.data:Vocabulary.build`
- `tests.test_model:test_config_round_trips_through_dict` → `src.model:ModelConfig.from_dict`
- `tests.test_split:test_book_split_keeps_books_out_of_each_others_splits` → `src.data:book_token_counts`
- `tests.test_split:test_book_split_keeps_books_out_of_each_others_splits` → `src.data:extract_epub_books`
- `tests.test_split:test_book_split_keeps_books_out_of_each_others_splits` → `src.data:split_token_ids_by_books`
- `tests.test_split:test_book_token_counts_sum_to_the_whole_corpus` → `src.data:book_token_counts`
- `tests.test_split:test_book_token_counts_sum_to_the_whole_corpus` → `src.data:extract_epub_books`
- `tests.test_split:test_book_token_counts_sum_to_the_whole_corpus` → `src.data:extract_epub_text`
- `tests.test_split:test_book_token_counts_sum_to_the_whole_corpus` → `src.data:tokenize`
- `tests.test_split:test_build_split_dataloaders_honours_explicit_splits` → `src.data:build_split_dataloaders`
- `tests.test_split:test_extract_epub_books_groups_chapters_by_book_opening` → `src.data:extract_epub_books`
- `tests.test_split:test_prepare_dataset_book_mode_records_the_splits` → `src.data:prepare_dataset`
- `tests.test_split:test_prepare_dataset_ratio_mode_remains_the_default` → `src.data:prepare_dataset`
- `tests.test_split:test_split_token_ids_by_books_rejects_a_count_mismatch` → `src.data:split_token_ids_by_books`
- `tests.test_split:test_split_token_ids_by_books_rejects_impossible_book_counts` → `src.data:split_token_ids_by_books`
- `tests.test_split:test_split_token_ids_by_books_takes_whole_books_contiguously` → `src.data:split_token_ids_by_books`
- `tests.test_tokenizer:test_encode_decode_round_trip` → `src.data:Vocabulary.build`
- `tests.test_tokenizer:test_tokenize_handles_multicharacter_punctuation` → `src.data:tokenize`
- `tests.test_tokenizer:test_tokenize_keeps_contractions_and_capitalization` → `src.data:tokenize`
- `tests.test_tokenizer:test_tokenize_separates_words_and_punctuation` → `src.data:tokenize`
- `tests.test_tokenizer:test_vocabulary_orders_by_frequency_then_first_appearance` → `src.data:Vocabulary.build`
- `tests.test_tokenizer:test_vocabulary_reserves_pad_and_unk_ids` → `src.data:Vocabulary.build`
- `tests.test_tokenizer:test_vocabulary_respects_max_size_and_maps_rare_words_to_unk` → `src.data:Vocabulary.build`
- `tests.test_tokenizer:test_vocabulary_save_and_load_round_trip` → `src.data:Vocabulary.build`
- `tests.test_tokenizer:test_vocabulary_save_and_load_round_trip` → `src.data:Vocabulary.load`
- `tests.test_training:test_checkpoint_records_metadata_for_standalone_evaluation` → `src.train:load_checkpoint`
- `tests.test_training:test_checkpoint_records_metadata_for_standalone_evaluation` → `src.train:save_checkpoint`
- `tests.test_training:test_checkpoint_round_trip_reproduces_predictions` → `src.train:load_checkpoint`
- `tests.test_training:test_checkpoint_round_trip_reproduces_predictions` → `src.train:save_checkpoint`
- `tests.test_training:test_clip_gradients_bounds_the_total_norm` → `src.train:clip_gradients_`
- `tests.test_training:test_overfit_one_batch_is_reproducible_with_a_seed` → `src.train:overfit_one_batch`
- `tests.test_training:test_overfit_one_batch_reduces_loss_substantially` → `src.train:overfit_one_batch`
- `tests.test_training:test_train_model_reports_each_epoch_as_it_finishes` → `src.train:train_model`
- `tests.test_training:test_train_model_saves_best_checkpoint_and_tracks_history` → `src.train:train_model`
- `tests.test_training:test_training_continues_when_no_log_callback_is_given` → `src.train:train_model`
