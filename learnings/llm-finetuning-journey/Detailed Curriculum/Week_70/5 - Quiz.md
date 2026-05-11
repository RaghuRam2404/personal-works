# Week 70 Quiz — Limitations, Polish, and Publication

## Multiple Choice

**Q1.** Which of the following is the best-written limitation sentence for your technical report?

A. "Our model may not work well on all SQL databases."
B. "Our model was evaluated only on PostgreSQL and TimescaleDB; accuracy on MySQL, Snowflake, and BigQuery is untested and may differ due to dialect-specific syntax differences in date arithmetic and window functions."
C. "Future work will address other SQL dialects."
D. "We acknowledge that our approach has limitations like all ML systems."

**Q2.** The NeurIPS reproducibility checklist asks: "Are all hyperparameters for the reported model results either reported or within the supplementary materials?" You report all hyperparameters except optimizer epsilon and AdamW beta2. Which response is most appropriate?

A. Delete the checklist item — it does not apply to fine-tuning
B. Answer "Yes" — epsilon and beta2 are standard defaults and are widely known
C. Answer "Partial" and add a footnote: "We used AdamW with standard defaults: β1=0.9, β2=0.999, ε=1e-8. Weight decay=0.01 for all stages."
D. Answer "No" and publish anyway without explanation

**Q3.** You finish writing your technical report and want to submit to arXiv. You do not have an institutional email address. What do you need to submit?

A. arXiv does not accept submissions without institutional email
B. An endorsement from an existing arXiv author in the cs.CL category
C. A paid arXiv Pro subscription
D. A co-author who has an institutional email

**Q4.** Your future work section says: "We plan to scale our approach to 70B models." A reviewer marks this as "too vague." What revision makes this future work statement concrete and evaluable?

A. "We plan to scale our approach to 70B models when compute becomes available."
B. "We plan to fine-tune Qwen2.5-72B on our v3 dataset using QLoRA (r=16) at 4-bit precision on 4× A100s, expecting 5–8 pp accuracy improvement on BIRD-SQL based on the Chinchilla scaling prediction."
C. "70B models will likely outperform 7B models on complex SQL tasks."
D. "Future work should investigate larger models."

## Short Answer

**Q5.** Your limitations section currently says "multi-turn SQL is a limitation." A co-author says this is too vague. Rewrite this limitation in 3 sentences that are specific, measurable, and actionable.

**Q6.** Explain why the evaluation prompt template must be included in the appendix verbatim (not paraphrased), and give one example of how a paraphrased template could produce different accuracy numbers.

**Q7.** You want to announce your report on Twitter with a 280-character summary. Write that announcement including: what the model does, the key result, and at least one link.

## Deep Scenario

**Q8.** Six months after publishing your report, a well-known ML researcher tweets: "The postgres-sqlcoder-7b results are not reproducible. I cannot get above 76% on Custom-200 using the released code and model." Your report claims 83.1%.

Write a 200-word response plan: (a) identify three possible explanations for the 7 pp gap, (b) describe what you will check in the first 24 hours, and (c) describe what you will publish as a correction or clarification.
