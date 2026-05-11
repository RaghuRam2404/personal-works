# Week 18 Quiz — Pretraining Data

## Multiple Choice

**Q1.** The key innovation of the RefinedWeb paper (Falcon team) was:

A) Mixing 22 high-quality data sources to achieve diversity  
B) Showing that aggressive filtering of a single source (Common Crawl) can match multi-source datasets  
C) Introducing the first deduplication technique for LLM pretraining  
D) Using GPT-4 annotations to score document quality  

---

**Q2.** MinHash deduplication is preferred over exact deduplication for large web corpora because:

A) MinHash is faster than exact hashing for all document sizes  
B) MinHash detects near-duplicates (same content, slight variations) that exact matching misses  
C) Exact deduplication is patented and cannot be used commercially  
D) MinHash works on document metadata without reading the full text  

---

**Q3.** FineWeb-Edu (1.3T tokens) performs better than FineWeb (15T tokens) on most academic benchmarks because:

A) Larger token counts always hurt model quality  
B) FineWeb uses a different base model for training  
C) FineWeb-Edu is filtered for educational quality, giving higher information density per token  
D) FineWeb contains too much code, which hurts language understanding  

---

**Q4.** When training a domain-specific SQL model, which filtering step is MOST likely to incorrectly remove useful SQL documents from a web corpus?

A) MinHash deduplication (Jaccard threshold 0.7)  
B) Alphabetic character ratio filter (alpha_ratio > 0.7)  
C) Language detection confidence filter (English > 0.65)  
D) Document word count filter (> 50 words)  

---

**Q5.** The Pile's mixing strategy (22 sources) contrasts with RefinedWeb's approach (single filtered source). Under which condition would you prefer The Pile's approach?

A) When you have limited compute and need fast data pipeline processing  
B) When the target domain requires diverse knowledge beyond what web text covers (e.g., biomedical, legal, code)  
C) When deduplication is too expensive to run across multiple sources  
D) When your model will be inference-only and not fine-tuned  

---

## Short Answer

**Q6.** Explain in 3–4 sentences why the duplicate_line_ratio filter (removing documents where > 30% of lines are repeated) is particularly effective at removing web boilerplate.

---

**Q7.** You are building a pretraining dataset for a PostgreSQL assistant. You download 10M SQL files from GitHub. Before training, list 4 specific quality filters you would apply (different from generic web text filters), and justify each.

---

**Q8.** A colleague says "our 50M-token SQL dataset is high-quality, so we don't need deduplication." Give 2 concrete reasons why they are wrong, with reference to specific training failure modes.

---

## Scenario

**Q9.** You are given 100GB of raw Common Crawl WARC files. You need to produce a clean 5GB English text dataset suitable for domain-adaptive pretraining of a SQL assistant. Design your filtering pipeline:

1. What tool do you use for text extraction from HTML?
2. What tool do you use for language detection, and what confidence threshold?
3. List 5 quality filters in order of application
4. What deduplication approach and settings (threshold, n-gram size)?
5. Estimate what fraction of 100GB survives to become the 5GB dataset (what is the expected compression ratio?)
