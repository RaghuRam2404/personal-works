# Week 18 Assignment Solutions

## Task 1 — Key Snippet: FineWeb Statistics

```python
from datasets import load_dataset
from collections import Counter
from urllib.parse import urlparse

ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True
)

stats = {"word_counts": [], "domains": [], "filter_rejections": Counter()}
for i, doc in enumerate(ds):
    if i >= 10000:
        break
    text = doc["text"]
    words = text.split()
    wc = len(words)
    stats["word_counts"].append(wc)

    url = doc.get("url", "")
    domain = urlparse(url).netloc
    stats["domains"].append(domain)

    lines = text.split("\n")
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    dup_lines = sum(1 for l in lines if lines.count(l) > 1) / max(len(lines), 1)

    if wc < 50: stats["filter_rejections"]["word_count<50"] += 1
    if dup_lines > 0.3: stats["filter_rejections"]["dup_line>0.3"] += 1
    if alpha_ratio < 0.7: stats["filter_rejections"]["alpha<0.7"] += 1
```

**Expected output (approximate values):**
- Mean document length: 500–1200 words (FineWeb-Edu skews longer)
- word_count < 50 rejection rate: ~3–8%
- dup_line > 0.3 rejection rate: ~5–12%
- alpha < 0.7 rejection rate: ~1–4%

**Common gotchas:**
- Using `.count(l)` inside a loop is O(N^2). For 10K docs it is fine; for 1M docs, use `Counter(lines)` instead
- The `url` field may be empty in some shards; handle with `doc.get("url", "")`
- Memory: streaming avoids OOM but you must buffer statistics separately, not store all docs

---

## Task 3 — Key Snippet: MinHash Deduplication

```python
from datasketch import MinHash, MinHashLSH

def get_shingles(text, n=5):
    words = text.lower().split()
    return set(" ".join(words[i:i+n]) for i in range(max(0, len(words)-n+1)))

def get_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for shingle in get_shingles(text):
        m.update(shingle.encode('utf8'))
    return m

lsh = MinHashLSH(threshold=0.7, num_perm=128)
signatures = {}

for doc_id, doc in enumerate(documents):  # documents is list of 5000 text strings
    m = get_minhash(doc["text"])
    signatures[doc_id] = m
    try:
        lsh.insert(f"d{doc_id}", m)
    except ValueError:
        pass  # duplicate key — shouldn't happen with unique ids

# Find duplicates
dup_count = 0
for doc_id, m in signatures.items():
    neighbors = lsh.query(m)
    if len(neighbors) > 1:  # more than itself
        dup_count += 1
```

**Expected output:**
- FineWeb-Edu duplicate rate: 0–3% (it is pre-deduplicated)
- Each MinHash insert: ~0.5ms at num_perm=128
- 5000 documents should complete in <30 seconds on Colab

**Common gotchas:**
- Empty documents crash `get_shingles` — add `if len(words) < n: return set()`
- LSH `threshold` is approximate; the actual Jaccard threshold varies by num_perm
- Do not call `lsh.query()` before inserting a document — it will return nothing useful

---

## Task 4 — How to Verify You Did It Right

```python
def quality_filter(doc):
    text = doc["text"]
    words = text.split()
    lines = [l for l in text.split("\n") if l.strip()]

    wc = len(words)
    if not (50 <= wc <= 100000):
        return False

    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.7:
        return False

    dup_ratio = sum(1 for l in lines if lines.count(l) > 1) / max(len(lines), 1)
    if dup_ratio >= 0.3:
        return False

    if len(lines) < 3:
        return False

    mean_line = sum(len(l) for l in lines) / max(len(lines), 1)
    if mean_line < 20:
        return False

    if any(len(l) > 1000 for l in lines):
        return False

    return True
```

**Expected pass rate on FineWeb-Edu:** 70–85% (it is already filtered; some documents still fail strict line-length filters).

**Red flags:**
- Pass rate > 98%: your filters are too loose or not applying correctly
- Pass rate < 50%: check that you are not accidentally running the filter on document IDs instead of text
- If `alpha_ratio` filter never fires: make sure you are dividing by `len(text)` not `len(words)`
