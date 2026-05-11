# Week 2 — Assignment

## Setup Checklist

- [ ] W&B is configured and `wandb login` works. Create a project called `week-02-makemore`.
- [ ] Spider dataset downloaded: `git clone https://github.com/taoyds/spider.git` or download from [yale-lily/spider](https://github.com/taoyds/spider). You only need the JSON files — no database access required this week.
- [ ] Karpathy's makemore videos queued: Parts 1, 2, and 3.
- [ ] Folder `week_02/` created in your `llm-finetuning-journey` repo.

---

## Task 1 — Bigram and MLP Language Models on Names

**Goal:** Build makemore from scratch following Karpathy's videos. Understand every line before moving to Task 2.

**Requirements:**
- Watch and code along with:
  - [Making makemore Part 1](https://www.youtube.com/watch?v=PaCmpygFfXo) — bigram model (lookup table approach).
  - [Making makemore Part 2](https://www.youtube.com/watch?v=TCH_1BHY58I) — MLP model with embeddings.
  - [Making makemore Part 3](https://www.youtube.com/watch?v=P6sfmUTpUmc) — activations, batch norm, initialization diagnostics.
- Save your implementations in:
  - `week_02/bigram_names.py`
  - `week_02/mlp_names.py`
- The MLP model must achieve **validation loss < 2.2** on the names dataset (Karpathy achieves ~2.17).
- Log train and validation loss to W&B project `week-02-makemore` every 500 steps.
- In `mlp_names.py`, add code to visualize the embedding space (2D projection using PCA or t-SNE). Plot which characters cluster together.

**Deliverable:** Committed files + W&B run link in `journal.md`.

---

## Task 2 — Swap Dataset to SQL Keywords

**Goal:** Adapt your MLP makemore model to generate SQL keyword sequences from the Spider dataset.

**Requirements:**
- Write `week_02/extract_sql_tokens.py` that:
  - Loads Spider's `train_spider.json`.
  - Extracts the `query` field from every example.
  - Tokenizes by splitting on whitespace and punctuation (simple split is fine — do not use HuggingFace tokenizers yet).
  - Keeps only alphabetic tokens (keywords like SELECT, FROM, WHERE, JOIN, GROUP, ORDER, etc.) and lowercases them.
  - Saves a vocabulary file `week_02/sql_vocab.txt` and a flat corpus file `week_02/sql_corpus.txt`.
- Write `week_02/mlp_sql.py` that:
  - Loads `sql_corpus.txt` as the training data (token sequences instead of character sequences).
  - Trains the same MLP architecture as Task 1, but on SQL tokens.
  - Context length: 4 previous tokens → predict the next token.
  - Runs for at least 5000 steps.
  - Logs to W&B project `week-02-makemore` (separate run, name it `sql-mlp`).
  - Generates 5 sample SQL keyword sequences from the trained model and prints them.

**Deliverable:** `week_02/extract_sql_tokens.py`, `week_02/mlp_sql.py`, 5 generated samples in `journal.md`. Commit message: `week-02-makemore`.

**Hints:**
- SQL keywords are a small vocabulary (~50–100 unique tokens). The embedding table will be tiny.
- The model will generate grammatically incorrect SQL — that is expected and fine. The exercise is about training loop mechanics and data pipeline, not SQL quality.
- Use `torch.multinomial` for sampling from the output distribution.

---

## Task 3 — Initialization Diagnostics

**Goal:** Reproduce Karpathy's activation and gradient statistics analysis (makemore Part 3) and understand what "dead neurons" look like.

**Requirements:**
- Take your `mlp_names.py` model.
- Before and after applying Kaiming init, compute and plot:
  - The mean and standard deviation of pre-activation values (before the nonlinearity) across the first 10 batches.
  - The gradient norms for each layer's weights at step 0.
- Save the plots as `week_02/activation_stats_before_kaiming.png` and `week_02/activation_stats_after_kaiming.png`.
- Write 3–5 sentences in `journal.md` explaining what you observe.

**Deliverable:** Two PNG files and the journal entry.

---

## Stretch Goals

- Add label smoothing to the cross-entropy loss in `mlp_names.py`. Use `nn.CrossEntropyLoss(label_smoothing=0.1)`. Does val loss improve? Why or why not for this task?
- Implement batch norm from scratch (as a class with `gamma`, `beta`, `running_mean`, `running_var`) without using `nn.BatchNorm1d`. Train with it. Verify your outputs match `nn.BatchNorm1d`.
- Try `nn.LayerNorm` in place of `nn.BatchNorm1d` and compare training stability. Note: batch norm requires a batch dimension; layer norm does not — this distinction matters a lot for transformers.
