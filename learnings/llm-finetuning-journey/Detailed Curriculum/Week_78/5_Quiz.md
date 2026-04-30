# Week 78 Quiz — Final Phase 6 Gate and Course Wrap

Difficulty: Senior research engineer. These questions assess your integrated understanding of the entire 18-month pipeline, your ability to evaluate your own work critically, and your readiness to extend or productize what you built.

---

## Multiple Choice

**Q1.** Your capstone model achieves 83.1% EM on Custom-200 (a domain benchmark you created) and 68.4% execution accuracy on BIRD-SQL dev. A recruiter at an ML team says "the Custom-200 result is impressive but we can't evaluate it externally." The most productive response is:

A. Publish the Custom-200 benchmark alongside your model so others can evaluate against it.
B. Explain that 68.4% on BIRD-SQL dev is a publicly verifiable result that places your model in the top-15 of the public leaderboard.
C. Concede the point — proprietary benchmarks are not useful for hiring evaluation.
D. Both A and B — publish Custom-200 for reproducibility and cite BIRD-SQL for external comparability.

---

**Q2.** You are writing the limitations section of your technical report. Which of the following is the most appropriate and useful limitation statement?

A. "The model may produce incorrect SQL in some cases."
B. "The model was not evaluated on dialects other than PostgreSQL and TimescaleDB; performance on MySQL syntax (e.g., backtick quoting, LIMIT without OFFSET) is unknown and expected to be lower due to training data composition."
C. "The model is a 7B parameter model and therefore smaller than GPT-4, which may limit its reasoning ability."
D. "As with all language models, this model may hallucinate."

---

**Q3.** You receive a request to fine-tune your model further on a client's 500-example proprietary SQL dataset. The client's queries involve Oracle syntax (ROWNUM, CONNECT BY). You have not done any Oracle-specific training. The most appropriate response is:

A. Run fine-tuning on the 500 Oracle examples using your current LoRA adapter; the model will adapt quickly because SQL is SQL.
B. Decline — your model is PostgreSQL-specialized and adding Oracle to a 7B model without CPT on Oracle data risks catastrophic interference.
C. Fine-tune on the 500 examples with a lower learning rate and a separate Oracle-specific LoRA adapter, test carefully on a held-out Oracle eval set, and be transparent with the client that the evaluation is preliminary.
D. Recommend GPT-4o fine-tuning instead because it handles multi-dialect SQL better.

---

**Q4.** Looking back at your full 18-month training pipeline (CPT → SFT-v3 → DPO-v3 → GRPO-final), you want to identify which stage contributed most to the TimescaleDB-specific advantage over GPT-4o. The cleanest ablation is:

A. Compare SFT-v3 checkpoint against GRPO-final checkpoint on Custom-200.
B. Compare base Qwen2.5-Coder-7B against SFT-v3 checkpoint, CPT-only checkpoint against SFT-v3, and SFT-v3 against DPO-v3 checkpoint — each comparison isolating one stage.
C. Run GPT-4o on your training data and see if its answers match yours.
D. Compare GRPO-final against a version without GRPO (i.e., DPO-v3 as the final model).

---

## Short Answer

**Q5.** You built Custom-200 as your primary evaluation benchmark. Identify two methodological weaknesses of Custom-200 as a benchmark, and for each, describe what you would need to do to address it in a rigorous research setting.

---

**Q6.** A junior engineer on your team asks why you used GRPO instead of PPO for the final alignment stage (Week 59). Give a 4–6 sentence answer that: (a) explains the core algorithmic difference, (b) explains why GRPO was the right choice for your specific setup, and (c) honestly states what GRPO's limitations are compared to PPO.

---

**Q7.** Describe the "curse of task specificity" as it applies to your capstone model. Your model achieves 83.1% EM on TimescaleDB queries but presumably lower performance on general SQL benchmarks like WikiSQL or Academic. Is this a problem? Justify your answer from both a research perspective and a product perspective.

---

## Deep Scenario

**Q8.** It is six months after completing this course. You have received an offer to join an ML team at a startup building an AI-powered BI tool (think: natural language queries over business dashboards backed by PostgreSQL). The role is "ML Engineer — NL2SQL." Your capstone work is directly relevant. In the interview, you are asked: "What are the top three technical challenges you would face taking your academic prototype to production, and how would you address each?"

Write a structured answer (6–8 sentences total) that covers three distinct challenges — one about data, one about evaluation, and one about deployment. For each challenge, name the problem specifically (not generically), describe the real-world consequence if unaddressed, and propose a concrete mitigation that you have actually implemented or studied during this course.

---

## Deep Scenario

**Q9.** Reflect on the following claim: "A 7B model that outperforms GPT-4o on a domain benchmark proves that task-specific fine-tuning of small models is the right approach for enterprise NLP." Evaluate this claim. Is it supported by your results? What does it miss? What conditions must be true for the claim to generalize beyond your specific case?

Write 5–7 sentences that engage with the claim seriously — neither dismissing it nor accepting it uncritically.
