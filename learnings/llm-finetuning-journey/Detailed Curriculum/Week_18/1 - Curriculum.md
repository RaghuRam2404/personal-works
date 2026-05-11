# Week 18 — Pretraining Data: Sources, Filtering, and Deduplication

## Learning Objectives

By the end of this week, you will be able to:

- Describe the lineage of major pretraining datasets: Common Crawl → C4 → The Pile → RefinedWeb → FineWeb
- Explain the key quality filtering steps applied to raw web crawl data
- Implement MinHash-based near-duplicate detection on a text corpus
- Run basic data statistics (length distribution, language identification) on FineWeb-Edu
- Articulate why data quality is often the decisive variable in modern LLM training

---

## Concepts

### The Raw Material: Common Crawl

[Common Crawl](https://commoncrawl.org/) is a non-profit that crawls the public web monthly and releases the data for free. As of 2024, the archive is over 250 petabytes. It is the primary source for virtually every modern pretraining dataset.

Common Crawl releases data in WARC format (Web ARChive). Each WARC file contains raw HTTP responses — HTML, headers, metadata. To get plain text, you need to:
1. Extract text from HTML (using tools like `trafilatura` or `resiliparse`)
2. Detect language (typically with `fasttext` or `langdetect`)
3. Apply quality filters
4. Deduplicate

The challenge: the raw crawl is mostly garbage. Boilerplate menus, spam, duplicate content, non-English text, and machine-generated text make up the majority of Common Crawl by document count.

### C4 (Colossal Clean Crawled Corpus)

C4 was introduced with the T5 paper (Raffel et al. 2020). It applies a set of heuristic quality filters to Common Crawl:
- Keep only lines ending with a terminal punctuation mark
- Remove pages with fewer than 3 sentences
- Remove pages containing "bad words" (keyword list)
- Remove duplicate 3-grams within a page (removes repetitive boilerplate)

C4 is 800GB of text. It became the standard reference dataset for several years.

**Limitation of C4:** the quality filters are very coarse. Many low-quality documents survive; many legitimate documents (code, lists, tables) are incorrectly removed.

### The Pile

The Pile ([arxiv.org/abs/2101.00027](https://arxiv.org/abs/2101.00027), Gao et al. 2021, EleutherAI) took a different approach: instead of filtering one source, mix many high-quality sources. The Pile's 22 sources include:

| Source | Size | Type |
|---|---|---|
| Pile-CC | 227GB | Filtered Common Crawl |
| PubMed Central | 90GB | Medical papers |
| Books3 | 100GB | Fiction and non-fiction |
| GitHub | 95GB | Code |
| Wikipedia | 6GB | Encyclopedia |
| ArXiv | 56GB | Academic papers |
| ... | ... | ... |

The mixing ratios were chosen manually based on training quality intuition. GPT-NeoX, GPT-J, and many other open models were trained on The Pile.

**Key insight from The Pile:** domain diversity matters. Models trained on diverse text generalize better than those trained on filtered web alone.

### RefinedWeb

RefinedWeb ([arxiv.org/abs/2306.01116](https://arxiv.org/abs/2306.01116), Penedo et al. 2023, Falcon) demonstrated that a single, aggressively filtered Common Crawl dataset can match or outperform The Pile's diverse mixture — if the filtering is done carefully enough.

RefinedWeb's key filtering steps:
1. URL filtering (block known spam, adult content domains)
2. Text extraction with `trafilatura`
3. Language detection with `fasttext`
4. Quality heuristics (repetition ratios, character ratios, line length statistics)
5. Exact deduplication with `ExactMatch`
6. Fuzzy deduplication with **MinHash**

The Falcon-40B and Falcon-180B models were trained on RefinedWeb. Their strong benchmark performance validated the "single source, aggressive filtering" approach.

### FineWeb

FineWeb ([HuggingFace dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb)) is HuggingFace's 2024 effort: 15T tokens from 96 Common Crawl dumps. FineWeb-Edu is a 1.3T-token subset filtered for educational quality using a classifier trained on GPT-4 annotations.

**You will download a 1GB shard of FineWeb-Edu for your assignment this week.** This is also the dataset you will use for tokenizer training in Week 20.

FineWeb's filtering pipeline (simplified):
1. URL deduplication across dumps
2. Language detection (English only)
3. `trafilatura`-based text extraction
4. Quality filters from the C4 + Gopher + RefinedWeb pipelines combined
5. MinHash fuzzy deduplication at 5-gram + Jaccard similarity threshold 0.7

### MinHash Deduplication — How It Works

Near-duplicate detection is critical: if 10% of your training data is duplicates, your model will memorize those documents and overfit to them. MinHash is the standard efficient algorithm for this.

**Step 1: Shingling.** Convert each document into a set of n-grams (typically 5-grams or 13-grams at the character or word level):

```python
def get_shingles(text, n=5):
    words = text.split()
    return set(" ".join(words[i:i+n]) for i in range(len(words)-n+1))
```

**Step 2: MinHash signatures.** For each document, compute k hash functions and take the minimum hash value from each. This gives a k-dimensional vector (the "MinHash signature"). The key property:

```
Pr[min_hash_i(A) == min_hash_i(B)] = |A ∩ B| / |A ∪ B| = Jaccard(A, B)
```

So the fraction of matching signature entries estimates Jaccard similarity.

```python
from datasketch import MinHash

def get_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for shingle in get_shingles(text):
        m.update(shingle.encode('utf8'))
    return m
```

**Step 3: Locality Sensitive Hashing (LSH).** Group documents into buckets by their signature. Documents that land in the same bucket are candidate duplicates. Check Jaccard similarity only within buckets (avoids O(N^2) comparisons).

```python
from datasketch import MinHashLSH

lsh = MinHashLSH(threshold=0.7, num_perm=128)
for doc_id, doc in enumerate(documents):
    m = get_minhash(doc)
    lsh.insert(f"doc_{doc_id}", m)

# Query for near-duplicates of a new document
result = lsh.query(get_minhash(new_document))
```

**Typical settings:**
- threshold = 0.7 (70% Jaccard similarity → near-duplicate)
- num_perm = 128 (trade-off between accuracy and memory)
- n-gram size = 5 words (too small → false positives; too large → misses near-dups)

### Quality Filtering Heuristics

Beyond deduplication, quality filtering is applied document-by-document. Common heuristics used in FineWeb and RefinedWeb:

| Filter | Threshold | Removes |
|---|---|---|
| Minimum word count | > 50 words | Very short pages |
| Maximum word count | < 100,000 words | Concatenated dumps |
| Fraction of lines starting with "#" | < 0.9 | Code-heavy pages |
| Fraction of duplicate lines | < 0.3 | Boilerplate |
| Fraction of alphabetic characters | > 0.7 | Mostly numeric/symbol pages |
| `fasttext` language confidence | > 0.65 | Uncertain language |

These filters are applied in sequence. Rejection at any stage removes the document from the dataset.

### Data Mixing

Modern LLM training datasets are not single-source. They mix:
- Web text (70–85%)
- Code (5–10%)
- Books (3–7%)
- Academic papers (2–5%)
- Curated/instructional data (1–3%)

Mixing ratios matter: Qwen2.5 emphasizes math and code; DeepSeek emphasizes code-first pretraining. You will study these mixing strategies in Week 24.

---

## Connections

**Prior week (17):** You now know how many tokens you need (Chinchilla). This week tells you where those tokens come from and how to trust them.

**Weeks 20–22:** You will train a 50M model on FineWeb-Edu samples. Your data pipeline from this week directly feeds that training.

**Weeks 25–26:** The domain dataset you build for PostgreSQL/TimescaleDB will use a subset of the filtering and format skills from this week.

---

## Common Misconceptions

- **"More data always means better quality."** False — 1B high-quality tokens can outperform 10B dirty tokens. FineWeb-Edu-1.3T beats FineWeb-15T on many benchmarks because of quality filtering.
- **"Exact deduplication is enough."** Near-duplicates (same article, different whitespace or one changed sentence) are abundant and cause memorization. MinHash catches these.
- **"fasttext language detection is perfect."** It reaches 70–80% precision on short documents and fails on code-mixed text. Always spot-check samples after filtering.
- **"I can use The Pile v1 for my project."** The Pile v1 has copyright issues (Books3 was removed). Use The Pile v2 or FineWeb instead.

---

## Time Allocation (6–8 hrs)

- 1h: Read The Pile paper (Sections 1–3, skim the rest)
- 1h: Read RefinedWeb paper (Sections 1–4)
- 1h: Read FineWeb dataset card + technical blog post
- 1.5h: Download FineWeb-Edu shard, run basic statistics
- 2h: Implement MinHash deduplication using `datasketch`
- 0.5h: Commit and write notes in `journal.md`
