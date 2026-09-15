# Harry Potter Word Predictor

A small, learning-first PyTorch project for comparing vanilla RNN, GRU, and LSTM word-level language models on one literary corpus.

> [!IMPORTANT]
> The Harry Potter books are copyrighted. This repository contains **no book text**. You must supply a lawfully obtained EPUB for personal research. The EPUB, extracted corpus, model checkpoints, and generated text are excluded by `.gitignore`.

## Learning question

**How does recurrent architecture affect next-word prediction on the same literary corpus?**

The project deliberately keeps the corpus, vocabulary, sequence split, training loop, and evaluation procedure fixed while changing the recurrent cell.

## Architecture

```text
harry-potter-rnn/
├── data/
│   ├── raw/                 # Local EPUB; never committed
│   └── processed/           # corpus.txt, vocab.json, token_ids.pt; never committed
├── src/
│   ├── __init__.py
│   ├── data.py              # EPUB extraction, cleaning, tokenization, vocabulary, datasets
│   ├── model.py             # Shared RNN/GRU/LSTM language-model class
│   ├── train.py             # One-batch check, training loop, checkpointing
│   ├── evaluate.py          # Split losses and perplexities
│   └── generate.py          # Top-k next words and recursive generation
├── tests/                   # Six learning-focused behavior tests
├── checkpoints/             # Local best checkpoints; never committed
├── results/                 # Local comparison tables/samples; never committed
├── .gitignore
├── requirements.txt
└── README.md
```

## Data flow

```text
lawfully obtained EPUB
  -> EPUB spine/reading-order extraction
  -> basic cleaning
  -> data/processed/corpus.txt
  -> word + punctuation tokens
  -> fixed vocabulary (<PAD>, <UNK>, then frequent tokens)
  -> contiguous 80/10/10 token split
  -> shifted fixed-length sequences
  -> RNN | GRU | LSTM
  -> loss, perplexity, next-word probabilities, generated text
```

The EPUB is parsed directly. Converting it to Markdown first is unnecessary and can introduce headings, links, and formatting artifacts into the training corpus.

## Planned command interface

```bash
# 1. Install dependencies
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Verify the selected GPU
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 3. Rebuild all processed data from the local EPUB
python -m src.data prepare \
  --input data/raw/harry-potter-series.epub \
  --output-dir data/processed \
  --vocab-size 18000 \
  --sequence-length 64

# 4. Inspect one shifted training batch
python -m src.data inspect --processed-dir data/processed --batch-size 4

# 5. Debug by overfitting one batch, then train
CUDA_VISIBLE_DEVICES=0 python -m src.train --model rnn --overfit-one-batch
CUDA_VISIBLE_DEVICES=0 python -m src.train --model rnn --epochs 15

# 6. Independently evaluate a saved checkpoint
python -m src.evaluate --checkpoint checkpoints/rnn-best.pt

# 7. Show probable next words
python -m src.generate \
  --checkpoint checkpoints/rnn-best.pt \
  --prompt "Harry looked at" \
  --top-n 10

# 8. Generate text
python -m src.generate \
  --checkpoint checkpoints/rnn-best.pt \
  --prompt "Harry opened the door" \
  --words 100 \
  --temperature 0.8 \
  --top-k 20
```

These commands define the intended interface. Implementation proceeds in tested stages; commands are not claimed to work until their stage is committed.

## Experiment order

1. Prepare and inspect the corpus.
2. Build and verify tokenization, vocabulary, contiguous splits, and shifted sequences.
3. Build the vanilla RNN and verify `[B, T] -> [B, T, V]` plus backpropagation.
4. Prove the model can overfit one batch.
5. Train and independently evaluate the RNN.
6. Add next-word prediction and generation.
7. Repeat unchanged training/evaluation for GRU, LSTM, and improved LSTM.
8. Compare parameter count, validation/test perplexity, training time, and generated text from one shared prompt.

## Minimal test contract

- Tokenizer: text → tokens → IDs → tokens.
- Dataset: every target equals its input shifted by one token.
- Model: `[4, 64]` produces `[4, 64, vocabulary_size]` for RNN, GRU, and LSTM.
- Backpropagation: trainable parameters receive gradients.
- One-batch overfit: loss falls substantially on a tiny fixed batch.
- Save/load: a reloaded checkpoint reproduces logits for the same input.

## Reproducibility rules

- `data/processed/` is rebuilt only from the local EPUB and explicit CLI settings.
- Splits are contiguous, not randomized, to avoid leaking nearby prose across splits.
- Checkpoints store model settings, vocabulary metadata, optimizer state, epoch, and validation loss.
- Model comparisons use the same processed corpus and split boundaries.
- Begin with `CUDA_VISIBLE_DEVICES=0`; distributed training is intentionally out of scope.

## Scope limitations

This is a word-level educational model, not a modern production language model. `<UNK>` collapses rare words, a 64-token context is short, and generated prose may memorize fragments. Do not publish the source corpus or long generated passages.
