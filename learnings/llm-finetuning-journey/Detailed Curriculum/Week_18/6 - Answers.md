# Week 18 Quiz Answers

## Q1 — Answer: B

**Answer:** B — Showing that aggressive filtering of a single source can match multi-source datasets.

**Why:** The RefinedWeb paper's central claim is that The Pile's diversity advantage evaporates when you filter Common Crawl aggressively enough. Falcon-40B trained on RefinedWeb matched or exceeded GPT-NeoX-20B (trained on The Pile) on many benchmarks, despite using only one source.

**Why others are wrong:**
- A describes The Pile's strategy, not RefinedWeb
- C: deduplication was used long before RefinedWeb; they refined it but did not invent it
- D: FineWeb used GPT-4 annotations; RefinedWeb used heuristic filters only

---

## Q2 — Answer: B

**Answer:** B — MinHash detects near-duplicates that exact matching misses.

**Why:** Web corpora contain the same news article reposted across 50 sites with minor formatting differences, the same Wikipedia paragraph quoted verbatim with one word changed, etc. Exact hash matching only finds byte-for-byte copies. MinHash's Jaccard similarity threshold catches these near-copies, which are the main source of memorization risk.

**Why others are wrong:**
- A: MinHash is actually slower than exact hashing on a per-document basis; its advantage is catching near-dups
- C: Exact deduplication is not patented
- D: MinHash must process the full text to compute shingles

---

## Q3 — Answer: C

**Answer:** C — FineWeb-Edu is filtered for educational quality, giving higher information density per token.

**Why:** A GPT-4-based classifier was trained to rate documents on an educational scale (1–5). Only documents scoring 3+ were included. This removes entertainment, clickbait, product reviews, and low-information content, leaving dense, factual, well-structured text that transfers better to academic benchmarks.

**Why others are wrong:**
- A: More tokens typically helps up to a point; the issue is quality, not quantity per se
- B: Both datasets use the same model; the difference is the training data subset
- D: FineWeb's code fraction is small; code does not hurt language benchmarks

---

## Q4 — Answer: B

**Answer:** B — Alphabetic character ratio filter (alpha_ratio > 0.7).

**Why:** SQL files are full of non-alphabetic characters: semicolons, parentheses, underscores, numbers, operators (`=`, `>`, `<`, `!=`). A typical SQL query might have an alpha ratio of 0.4–0.6, well below the 0.7 threshold used for natural language text. This filter, designed to remove spam and symbolic garbage, would also remove legitimate SQL content.

**Why others are wrong:**
- A: MinHash at 0.7 Jaccard would only remove near-identical queries, which is fine
- C: SQL documentation is usually English; language detection would keep it
- D: SQL queries are often short but rarely below 50 words in documentation pages

---

## Q5 — Answer: B

**Answer:** B — When the target domain requires diverse knowledge beyond web text (biomedical, legal, code).

**Why:** Common Crawl, even filtered, skews toward general English web content. If your model needs deep biomedical knowledge, PubMed Central is a far better source than hoping biomedical content survives web filtering. The Pile's deliberate inclusion of domain-specific sources addresses knowledge gaps that web filtering cannot fix.

**Why others are wrong:**
- A: A multi-source pipeline is more complex, not less — opposite of fast
- C: Deduplication across multiple sources is more work but is done routinely
- D: Data source choice is about training, not inference

---

## Q6 — Short Answer

Web pages often contain navigation menus, footers, cookie notices, and boilerplate site templates that repeat identically across hundreds of documents. When a document is a forum page or e-commerce listing, the surrounding template (header, footer, sidebars) can constitute 50–70% of the extracted text. These repeated lines drive up the duplicate_line_ratio. By filtering documents where > 30% of lines repeat within the document, you specifically target these template-heavy pages while leaving genuine article text (where few lines repeat verbatim) intact.

---

## Q7 — Short Answer (SQL-specific filters)

1. **Syntax validity filter**: Run a lightweight SQL parser (e.g., `sqlglot.parse(sql, raise_on_error=False)`) and discard files that produce syntax errors for > 50% of statements. Rationale: corrupted or auto-generated invalid SQL is noise.

2. **Comment-to-code ratio**: Keep files where at least 10% of lines are comments. Rationale: commented SQL files are likely hand-written and higher quality; pure SQL without any explanation is harder for the model to learn from.

3. **Table name diversity filter**: Count distinct table names referenced. Discard files that only reference generic names like `table1`, `t1`, `test`. Rationale: these are likely tutorial boilerplate, not real-world queries.

4. **PostgreSQL-specific keyword filter**: Boost (or filter in) files containing at least one PostgreSQL-specific keyword: `RETURNING`, `ON CONFLICT`, `JSONB`, `TSTZRANGE`, `pg_`, etc. Rationale: generic ANSI SQL does not teach PostgreSQL idioms.

---

## Q8 — Short Answer

First, even a manually curated 50M-token dataset likely has near-duplicates from copy-paste across projects (the same boilerplate `CREATE TABLE` schema copied across 500 repos). Without deduplication, the model will memorize these patterns disproportionately, producing output that reflects the most common boilerplate rather than diverse query styles.

Second, during fine-tuning, duplicate examples act as artificial up-weighting of specific (instruction, output) pairs. If 20% of your 50M tokens is variations of the same 10 table schemas, the model will overfit to those schemas and generalize poorly to new table structures your SQL assistant will encounter in production.

---

## Q9 — Scenario Model Answer

1. **Text extraction:** `trafilatura` — it is the current best practice for precision text extraction from HTML, outperforming `BeautifulSoup` and `boilerpy3` on web content.

2. **Language detection:** `fasttext` with `lid.176.bin`, confidence threshold > 0.65. Use 0.65 (not 0.9) to avoid discarding legitimate English pages with technical jargon or mixed language names.

3. **Quality filters in order:**
   - URL blocklist (remove known spam/adult domains — fast, cheap)
   - Minimum word count > 50 (remove stubs)
   - Alpha ratio > 0.65 (remove symbol-dominated pages — lower than default since web content can include tables)
   - Duplicate line ratio < 0.3 (remove boilerplate)
   - Mean line length > 20 chars (remove navigation menus)

4. **Deduplication:** MinHash LSH with Jaccard threshold = 0.7, num_perm = 128, 5-word shingles. Run within each domain group first, then globally.

5. **Expected compression ratio:** Raw Common Crawl text extraction typically retains 40–50% of WARC bytes. Language + quality filters typically remove 60–70% of extracted text. Deduplication removes another 20–40%. Overall: 100GB × 0.45 × 0.35 × 0.7 ≈ 11GB — so a 100GB raw crawl producing 5GB clean text implies a ~5% retention rate, which is typical.
