# Week 18 TakeAway — Pretraining Data

**One-liner:** High-quality filtered Common Crawl beats diverse dirty data; MinHash catches near-duplicates that destroy training.

---

## Key Pipeline

```
Raw WARC → trafilatura (HTML→text) → fasttext (lang detect ≥ 0.65)
→ heuristic quality filters → exact dedup → MinHash LSH (J ≥ 0.7)
→ clean text ready for tokenizer
```

---

## Key Code Patterns

```python
# MinHash dedup — production pattern
from datasketch import MinHash, MinHashLSH

def get_minhash(text, num_perm=128, n=5):
    words = text.lower().split()
    m = MinHash(num_perm=num_perm)
    for i in range(len(words) - n + 1):
        m.update(" ".join(words[i:i+n]).encode("utf8"))
    return m

lsh = MinHashLSH(threshold=0.7, num_perm=128)
# Insert: lsh.insert(key, minhash)
# Query:  lsh.query(minhash)  → list of near-duplicate keys
```

```python
# Quality filter — minimal production set
def quality_filter(text):
    words = text.split()
    if not (50 <= len(words) <= 100_000):
        return False
    alpha_r = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_r < 0.7:
        return False
    lines = [l for l in text.split("\n") if l.strip()]
    dup_r = sum(lines.count(l) > 1 for l in lines) / max(len(lines), 1)
    if dup_r > 0.3:
        return False
    return True
```

---

## Decision Rules

- Use `trafilatura` for HTML extraction (not BeautifulSoup)
- Use `fasttext lid.176.bin` for language detection; threshold 0.65 for web, 0.80 for curated
- Run exact dedup before MinHash (exact is cheaper; removes easy cases first)
- MinHash settings: threshold=0.7, num_perm=128, n=5 word shingles — standard for pretraining
- Alpha ratio threshold: 0.7 for web text; lower (0.5) if including code or SQL
- If building SQL dataset: disable alpha filter or lower to 0.4

---

## Numbers to Remember

| Dataset | Size | Source |
|---|---|---|
| C4 | 800GB | Common Crawl (T5 filters) |
| The Pile v2 | 825GB | 22 sources mixed |
| RefinedWeb | 600B tokens | CC + aggressive filters |
| FineWeb | 15T tokens | 96 CC dumps |
| FineWeb-Edu | 1.3T tokens | FineWeb + edu classifier |

- Common Crawl text extraction retains ~40–50% of WARC bytes
- Quality + language filters remove ~60–70% of extracted text
- MinHash dedup removes ~20–40% more (varies by domain)
- End-to-end: ~5–15% of raw WARC becomes clean training data

---

## Red Flags

- Duplicate rate = 0% on raw web data → your MinHash is broken
- Alpha filter removing > 30% of docs → likely too aggressive; check for SQL/code content
- fasttext confidence threshold at 0.9 → removes too many legitimate short documents
- Not running dedup at all → model memorizes repeated documents, fails perplexity eval
