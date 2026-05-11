# Week 18 Glossary — Pretraining Data

**Common Crawl**: A non-profit web archive containing petabytes of raw HTML from monthly web crawls; the source of most pretraining datasets.

**WARC (Web ARChive)**: The file format used by Common Crawl; contains raw HTTP responses including HTML, headers, and metadata.

**C4 (Colossal Clean Crawled Corpus)**: A 800GB filtered Common Crawl dataset introduced with T5; uses heuristic quality filters on raw text.

**The Pile**: An 825GB diverse pretraining dataset from EleutherAI combining 22 sources including web, books, academic papers, and code.

**RefinedWeb**: A 600B-token Common Crawl dataset from the Falcon team showing that aggressive filtering of a single source can match multi-source quality.

**FineWeb**: HuggingFace's 15T-token dataset from 96 Common Crawl dumps with combined quality filters from C4, Gopher, and RefinedWeb.

**FineWeb-Edu**: A 1.3T-token educational quality subset of FineWeb, filtered by a GPT-4-trained educational quality classifier.

**trafilatura**: A Python library for extracting main body text from HTML; outperforms BeautifulSoup for web content extraction.

**fasttext**: A Facebook library for text classification; widely used for language detection via the `lid.176.bin` model.

**MinHash**: A locality-sensitive hashing technique for estimating Jaccard similarity between documents in sub-linear time.

**Jaccard similarity**: |A ∩ B| / |A ∪ B| — the fraction of shared n-grams between two documents; used as the duplicate threshold.

**Shingling (n-gram shingles)**: Converting a document into a set of overlapping n-gram strings; the input representation for MinHash.

**LSH (Locality Sensitive Hashing)**: A bucketing technique that groups similar MinHash signatures together for efficient nearest-neighbor search.

**Duplicate line ratio**: The fraction of lines in a document that appear more than once; a proxy for boilerplate and template content.

**Alpha ratio**: The fraction of alphabetic characters in a document; used to filter out mostly-numeric or mostly-symbolic pages.

**Data mixing**: The practice of combining multiple source datasets at specific ratios during pretraining to balance knowledge domains.

**Quality filtering**: Removing low-quality documents from a corpus using rule-based heuristics before or after deduplication.
