# Week 18 Assignment — Pretraining Data Analysis and Deduplication

## Setup Checklist

- [ ] `pip install datasets datasketch fasttext-wheel langdetect trafilatura`
- [ ] HuggingFace account and `huggingface_hub` configured (`huggingface-cli login`)
- [ ] Colab Free notebook OR local environment with ~4GB RAM available
- [ ] GitHub repo with `week-18-pretraining-data/` directory

---

## Task 1 — Download and Analyze a FineWeb-Edu Shard

**Goal:** Get hands-on with a real pretraining dataset shard and compute basic statistics.

**Requirements:**

Download a 1GB shard of the FineWeb-Edu dataset using the HuggingFace datasets library. Do NOT download the full dataset.

```python
from datasets import load_dataset
ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",    # 10B-token sample, much smaller than full
    split="train",
    streaming=True         # stream — do not download all at once
)
```

From the first 10,000 documents, compute and report:
- Total token count (use `len(text.split())` as a word proxy, then multiply by ~1.3 for BPE tokens)
- Document length histogram (buckets: <100, 100–500, 500–2000, 2000–5000, >5000 words)
- Fraction of documents that would be removed by each of these filters:
  - `word_count < 50`
  - `duplicate_line_ratio > 0.3` (lines that appear more than once / total lines)
  - `alpha_ratio < 0.7` (alphabetic chars / total chars)
- Top 20 domains by frequency (the `url` field contains this)

Save as a Markdown report: `week-18-pretraining-data/data_analysis.md`

**Deliverable:** `data_analysis.md` with all statistics, plus the analysis notebook.

**Acceptance criteria:** Your report includes all 4 statistics above with real numbers from the actual dataset, not placeholder values.

---

## Task 2 — Language Detection with fasttext

**Goal:** Identify what fraction of raw Common Crawl passes language filtering.

**Requirements:**

- Download the fasttext language detection model (`lid.176.bin`)
- Sample 1,000 documents from a raw Common Crawl-adjacent dataset (use `mc4` as a proxy since raw CC requires more setup):

```python
ds_raw = load_dataset("allenai/c4", "en", split="train", streaming=True)
```

- For each document in a 1,000-document sample, run language detection
- Report: fraction that is English with confidence > 0.65, fraction that is English with confidence > 0.90, top 5 languages found

**Deliverable:** Add a "Language Detection" section to `data_analysis.md`.

**Hints:**
- `fasttext` model returns `((label, confidence),)` — the label is `__label__en`
- Very short documents will have low confidence; this is expected
- `pip install fasttext-wheel` is the maintained fork for Python 3.10+

---

## Task 3 — Implement MinHash Deduplication

**Goal:** Build a working near-duplicate detector using the `datasketch` library.

**Requirements:**

1. Take 5,000 documents from FineWeb-Edu (streaming, first 5K)
2. Implement `get_shingles(text, n=5)` → returns a set of 5-gram strings
3. Implement `get_minhash(text, num_perm=128)` → returns a `datasketch.MinHash` object
4. Build an `MinHashLSH` index with `threshold=0.7`
5. Insert all 5,000 documents
6. For each document, query the LSH for near-duplicates
7. Report:
   - Total documents: 5,000
   - Documents with at least 1 near-duplicate: N
   - Estimated duplicate rate: N / 5000
   - Show 2 example near-duplicate pairs with their Jaccard similarity

**Deliverable:** `week-18-pretraining-data/minhash_dedup.py` + output printed to stdout.

**Hints:**
- If the duplicate rate is exactly 0%, your threshold may be too high or your shingles too large — try `n=3` and `threshold=0.5` on a test
- `MinHash.jaccard(m1, m2)` returns the estimated Jaccard similarity between two MinHash objects
- FineWeb-Edu is already deduplicated, so expect a low rate (<5%); the exercise is to build the pipeline, not to find many duplicates

---

## Task 4 — Quality Filtering Pipeline

**Goal:** Write a reusable quality filter that mirrors the FineWeb approach.

**Requirements:**

Write a function `quality_filter(doc: dict) -> bool` that returns `True` if the document should be kept. Apply all of these filters:
- `word_count >= 50`
- `word_count <= 100000`
- `alpha_ratio >= 0.7`
- `duplicate_line_ratio < 0.3`
- `mean_line_length >= 20` (average characters per line)
- No line exceeds 1000 characters AND the document has at least 3 lines

Apply this filter to your 5,000-document sample. Report:
- Overall pass rate
- Which filter rejects the most documents (compute per-filter rejection count)

**Deliverable:** Add to `minhash_dedup.py` or create `quality_filter.py`.

GitHub commit: `week-18-pretraining-data`

---

## Stretch Goals

- Run the same pipeline on a small Code dataset (`bigcode/the-stack-smol`) and compare duplicate rates vs. FineWeb-Edu
- Implement exact URL deduplication in addition to MinHash (exact URL matching to remove documents from the same page seen in multiple crawl dumps)
- Benchmark `datasketch` MinHash speed on 50K documents and extrapolate: how many CPU-hours would deduplicating 15T tokens of FineWeb take?
