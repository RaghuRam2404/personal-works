# Week 70 Assignment — Technical Report: Limitations, Appendix, and Publish

## Setup Checklist

- [ ] `report/report_draft_v2.md` (through Section 6 from Week 69) ready
- [ ] All HuggingFace Hub repos public and linked
- [ ] arXiv account created (or decision made to publish as HF model card PDF)
- [ ] LaTeX or Pandoc + XeLaTeX installed for PDF generation

## Task 1: Write Limitations and Future Work

**Goal:** An honest, specific, and research-forward Section 7.

**Requirements:**
- [ ] Section 7.1 Limitations: 5 specific limitations, each with: what the limitation is, what its practical impact is, and whether any mitigation exists
- [ ] Do not include vague limitations ("results may not generalize") — every limitation must be testable or measurable
- [ ] Section 7.2 Future Work: 3 specific future directions, each with: the specific action, the mechanism by which it is expected to help, and the success metric
- [ ] Total word count for Section 7: 300–500 words
- [ ] At least one limitation must be a calibration or safety concern (not just accuracy)

**Deliverable:** `report/limitations_section.md`

## Task 2: Build the Appendix

**Goal:** A reproducibility appendix that passes the NeurIPS checklist.

**Requirements:**
- [ ] Appendix A: Full hyperparameter table (copy from `report/hyperparams_table.md`, add any missing values: optimizer betas, epsilon, weight decay, gradient clipping)
- [ ] Appendix B: Compute budget table (stage | GPU type | GPUs | hours | total GPU-h | approx cost)
- [ ] Appendix C: Data release (dataset name, HF Hub link, license, number of examples per split)
- [ ] Appendix D: Model release (all 4 repos: BF16, Q4_K_M GGUF, AWQ INT4, GPTQ INT4 — with links and licenses)
- [ ] Appendix E: Evaluation prompt template (exact text used for all benchmark evaluations, copied from your eval script)
- [ ] Appendix F: Custom-200 benchmark description (how examples were written, expert qualifications, schema sources, time period)
- [ ] Walk through the NeurIPS reproducibility checklist; for each "No" answer, add a footnote explaining why

**Deliverable:** `report/appendix.md`

## Task 3: Final Integration and Polish

**Goal:** A single, polished `report/final_report.md` document.

**Requirements:**
- [ ] Assemble all sections: Abstract + 1. Introduction + 2. Related Work (brief, or omit with a note) + 3. Dataset + 4. Training + 5. Evaluation + 6. Ablations + 7. Limitations + References + Appendices
- [ ] Consistency pass: run `grep` for every key number (83.1, 25500, 102M, etc.) — each must appear consistently throughout
- [ ] Citation pass: every cited work has author + year + arXiv or venue; no "et al." for 2-author papers
- [ ] Table captions: every table has a caption and is referenced in the body text
- [ ] Tense pass: methods past tense, section overview present tense
- [ ] Target: 3,500–5,500 words total for the main body

**Deliverable:** `report/final_report.md`

## Task 4: Generate PDF and Publish

**Goal:** A publicly available preprint.

**Requirements:**
- [ ] Generate PDF: `pandoc report/final_report.md -o report/postgres-sqlcoder-7b-report.pdf --pdf-engine=xelatex -V geometry:margin=1in`
- [ ] Upload PDF to your model's HuggingFace Hub repository under `report/postgres-sqlcoder-7b-report.pdf`
- [ ] If submitting to arXiv: prepare LaTeX source, submit to cs.CL, record the arXiv ID
- [ ] Write a 3-sentence announcement post (for Twitter/LinkedIn/HF blog): title, key result, links to HF repos and report
- [ ] Post the announcement to at least one public channel

**Deliverable:** Public PDF link + announcement post text saved as `report/announcement.md`

## Stretch Goals

- Write a full Related Work section (500 words, 3 clusters: text-to-SQL models, instruction tuning methods, RLHF/RLAIF approaches)
- Create a visual abstract (one-page diagram showing the full pipeline: CPT → SFT → DPO → GRPO → quantization → deployment)
- Submit to HuggingFace Daily Papers for community visibility
