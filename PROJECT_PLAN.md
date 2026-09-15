# Staged implementation plan

Each stage follows: explain → test first → implement the smallest version → run → show output → commit.

| Commit | Working stage | Verification gate |
|---|---|---|
| 1 | Repository architecture and data safeguards | Ignored sample EPUB/corpus do not appear in Git |
| 2 | EPUB extraction and basic corpus cleaning | Fixture EPUB becomes deterministic clean text |
| 3 | Tokenizer and vocabulary | Text round-trip works, including `<UNK>` |
| 4 | Contiguous splits and shifted datasets | Target equals input shifted by one token |
| 5 | Shared recurrent model with vanilla RNN | Shape and backward tests pass |
| 6 | One-batch diagnostic and training | Tiny batch loss falls; checkpoint reload is identical |
| 7 | Independent evaluation | Loss and `exp(loss)` perplexity reported per split |
| 8 | Prediction and generation | Top probabilities sum correctly; seeded generation repeats |
| 9 | GRU and LSTM comparison | Same pipeline works for all recurrent cells |
| 10 | Improved LSTM and final comparison | Shared report includes parameters, PPL, time, and samples |

Full-corpus training begins only after the local EPUB is available and all CPU-scale tests pass.
