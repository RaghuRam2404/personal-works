# Week 67 Assignment — Technical Report: Outline, Abstract, Introduction, Dataset

## Setup Checklist

- [ ] Create `report/` directory in your project repo
- [ ] Reference papers downloaded (PDFs): Llama 2, Tulu 2, SQLCoder model card
- [ ] W&B dashboards from Weeks 53–60 open (for exact dataset and training numbers)
- [ ] Your evaluation results from Weeks 61–62 available (accuracy tables)
- [ ] Markdown editor ready (VSCode with Preview, Obsidian, or Typora)

## Task 1: Full Report Outline

**Goal:** A skeleton that all four writing weeks will fill in, with section headers and one-line descriptions of each subsection's content.

**Requirements:**
- [ ] Create `report/report_outline.md` with all 8 sections and all subsections
- [ ] Each subsection has a one-line note on what numbers/content goes there
- [ ] Section word-count targets: Abstract 150–200w, Introduction 400–600w, Dataset 600–900w, Training 600–900w, Evaluation 600–900w, Ablations 300–500w, Limitations 200–400w, Appendix 300–500w
- [ ] Total target: 3,000–5,000 words for the complete report

**Deliverable:** `report/report_outline.md`

## Task 2: Write the Abstract

**Goal:** A 150–200 word abstract that accurately summarizes your contribution, method, and results.

**Requirements:**
- [ ] Four structural components: (1) problem + gap, (2) your approach, (3) key results (actual numbers), (4) what you release
- [ ] Contains at least three numbers from your actual evaluation results
- [ ] Does not contain hedging language ("we believe", "we hope")
- [ ] Write three draft abstracts, then pick the best and revise it once more
- [ ] Save as `report/abstract_draft.md` with all three drafts labeled

**Deliverable:** `report/abstract_draft.md` with three drafts + final selected abstract

## Task 3: Write the Introduction

**Goal:** A 400–600 word introduction that motivates the problem, identifies the gap, and lists contributions.

**Requirements:**
- [ ] Paragraph 1: the problem (text-to-SQL for production PostgreSQL, not benchmark SQL)
- [ ] Paragraph 2: the gap (closed-source cost/latency + open-weight models not domain-adapted)
- [ ] Paragraph 3: your contributions as a bulleted list (5 items)
- [ ] Paragraph 4: paper overview ("Section 3 describes...")
- [ ] At least 3 inline citations (author, year format) to related work
- [ ] No fluff — every sentence carries a claim or a citation

**Deliverable:** `report/introduction.md`

## Task 4: Write the Dataset Section

**Goal:** A 600–900 word dataset section with enough detail for reproduction.

**Requirements:**
- [ ] Section 3.1: Continued pretraining corpus — source, size (tokens), processing steps, deduplication method
- [ ] Section 3.2: SFT Dataset v3 — three subsections (curated, synthetic, adapted), sizes for each, total after filtering
- [ ] Section 3.3: DPO preference dataset — construction method, acceptance criteria, pair counts
- [ ] Section 3.4: Quality filtering — judge model, threshold, acceptance rate, deduplication cutoff
- [ ] One example data point (schema snippet + question + SQL) presented as a figure
- [ ] One table: dataset statistics (source, size, format, filtering applied)
- [ ] All numbers must match your W&B logs from Weeks 53–55

**Deliverable:** `report/dataset_section.md`

## Stretch Goals

- Write a detailed related work section (1–2 paragraphs per related work cluster: text-to-SQL benchmarks, open-weight SQL models, instruction tuning methods, RLHF/RLAIF methods)
- Create a dataset statistics visualization (SQL complexity distribution, schema size distribution) using matplotlib and save as `report/figures/dataset_stats.png`
- Draft a data release statement: what you will release, under what license, where it will be hosted
