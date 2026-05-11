# Week 70 Assignment Solutions

## Task 1: Limitations Section Template

```markdown
## 7. Limitations and Future Work

### 7.1 Limitations

**Single-dialect coverage.** The model was trained exclusively on PostgreSQL syntax.
MySQL and Snowflake use different date arithmetic, LIMIT syntax, and window function
conventions. Accuracy on these dialects is unknown. A researcher targeting multi-dialect
SQL should fine-tune a separate adapter for each dialect.

**Benchmark coverage gaps.** Custom-200 covers time-series aggregations well but
under-represents recursive CTEs, lateral joins, PostGIS functions, and full-text search
(`tsvector`/`tsquery`). Accuracy on these patterns should be treated as unknown.

**Single-turn only.** The model is trained on isolated question-SQL pairs. Multi-turn
correction workflows ("the previous query was wrong because the date filter is missing")
are not characterized. Section 8.2 proposes a direction for this.

**Quantization accuracy gap.** The Q4_K_M GGUF variant shows a 2.1 pp accuracy drop
vs BF16 (81.0% vs 83.1% on Custom-200). For production systems requiring maximum
accuracy at low latency, this tradeoff must be evaluated per-deployment context.

**No calibration study.** The model's log-probability confidence is not validated as
a predictor of correctness. Overconfident wrong SQL poses safety risks in automated
execution pipelines. Users should validate model confidence before enabling auto-execution.

### 7.2 Future Work

**Multi-turn SQL refinement.** Fine-tune on CoSQL and a custom multi-turn dataset
to teach the model to accept a previous SQL + error message + corrected question as
context. Expected mechanism: CoSQL's turn-level annotations teach the model to
distinguish user intent change from user correction. Success metric: improvement on
CoSQL dev execution accuracy from the current baseline.

**Bilingual NL→SQL (English + Tamil).** Expand the SFT dataset with Tamil natural
language questions paired with English SQL answers. This addresses enterprise users
in South Indian markets. Success metric: parity with English accuracy on a 50-example
Tamil→SQL test set.

**TimescaleDB continuous aggregate support.** Current training data rarely includes
`CREATE MATERIALIZED VIEW ... WITH (timescaledb.continuous)` patterns. Add 500+
training examples for continuous aggregate creation and refresh queries. Expected
5–10 pp improvement on the "continuous aggregates" subcategory of Custom-200.
```

## Task 2: Appendix Reproducibility Checklist

NeurIPS items to address:

| Checklist Item | Status | Note |
|---|---|---|
| Training code released | Yes | GitHub repo linked in Section 4 |
| Evaluation code released | Yes | `eval/run_eval.py` in repo |
| All hyperparameters reported | Yes | Appendix A Table |
| Compute budget reported | Yes | Appendix B |
| Dataset released | Yes | HF Hub link in Appendix C |
| Model weights released | Yes | 4 HF Hub repos in Appendix D |
| Test set released | Yes | Custom-200 in Appendix F |

Any "No" item: add a brief sentence explaining the constraint (e.g., "Closed-source API baselines cannot be released; evaluation was conducted via official APIs using published endpoint names and versions").

## Task 3: Final Consistency Pass Script

```bash
# Check all key numbers appear consistently
for num in "83.1" "25,500" "102M" "5,000" "1,534" "1,034" "4.2"; do
    echo "=== $num ==="
    grep -n "$num" report/final_report.md
done
```

Every number should appear in the same form throughout. If you see "25500" in one place and "25,500" in another, pick one format and apply it everywhere.

## Task 4: PDF and Announcement

```bash
pandoc report/final_report.md \
    -o report/postgres-sqlcoder-7b-report.pdf \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    --toc
```

Announcement post template:
```
postgres-sqlcoder-7b: a 7B open-weight model for PostgreSQL + TimescaleDB SQL,
trained via CPT → SFT → DPO → GRPO, achieving 83.1% on our domain benchmark —
surpassing GPT-4o (79.4%) and running locally in 4.5 GB.

Models: [HF link to all 3 quantized variants]
Report: [arXiv or HF PDF link]
Code + data: [GitHub link]
```

## Common Gotchas

- The appendix prompt template must be the exact string used in evaluation, not a cleaned-up version. Copy-paste from your eval script.
- PDF generation with `pandoc` and markdown tables sometimes breaks on complex tables; test early and fix formatting before the deadline.
- arXiv submission requires LaTeX for acceptance at most venues; if you write in Markdown, convert to LaTeX using Pandoc before arXiv submission.
- Do not upload a password-protected or scanned PDF to HuggingFace — ensure text is selectable and searchable.

## How to Verify You Did It Right

The ultimate test: give the report PDF to a colleague who has not seen your code. Ask them: "Can you reproduce my SFT dataset from Section 3?" and "Can you reproduce my training from Section 4?" If they say yes to both, you have written a reproducible technical report.
