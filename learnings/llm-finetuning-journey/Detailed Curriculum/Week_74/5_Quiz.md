# Week 74 Quiz — Context Extension: LongRoPE and YaRN

## Multiple Choice

**Q1.** A model trained with RoPE at 4096 tokens is given a 8192-token input without any context extension. What is the primary failure mode?

A. The attention computation runs out of memory due to the longer sequence
B. Position indices 4097–8192 produce rotation angles outside the training distribution, causing attention to degrade on tokens in the extended range
C. The model generates repetitive output because its positional encoding wraps around at 4096
D. Flash attention does not support sequences longer than the training context

**Q2.** YaRN applies different scaling factors to different RoPE frequency dimensions. Specifically, it does NOT interpolate the high-frequency dimensions. Why?

A. High-frequency dimensions are not used in standard attention computation
B. High-frequency dimensions encode long-range relationships that can be safely compressed
C. High-frequency dimensions encode short-range positional distinctions; interpolating them degrades the model's ability to distinguish nearby tokens
D. High-frequency dimensions correspond to the first few layers and are frozen during fine-tuning

**Q3.** LongRoPE achieves better perplexity at 128K context than YaRN using the same fine-tuning budget. What is the computational cost that LongRoPE pays for this improvement?

A. LongRoPE requires 10x more fine-tuning steps than YaRN
B. LongRoPE's evolutionary search requires hundreds of candidate evaluations, each requiring a short fine-tuning run — total cost is 50–200 GPU-hours
C. LongRoPE uses a larger model (13B vs 7B) to achieve the improvement
D. LongRoPE requires storing the full evolutionary search history in GPU memory during training

**Q4.** Your SQL model needs to handle a 6000-token schema. Instead of context extension, your team proposes "schema compression" — automatically selecting only the relevant tables from the schema based on the question. What is the primary failure risk of this approach?

A. Schema compression is computationally too expensive for real-time use
B. If the compression step misidentifies the relevant tables, the SQL generation model receives incomplete schema information and will fail on queries requiring the omitted tables
C. Schema compression requires a separate model that is as large as the SQL model itself
D. Schema compression is incompatible with TimescaleDB hypertable schemas

## Short Answer

**Q5.** Explain the difference between "position interpolation" and "NTK-aware scaling" for extending RoPE context length. What problem does each approach solve, and which problem remains after position interpolation that NTK-aware scaling addresses?

**Q6.** Your model was trained at 4096 tokens. You apply YaRN with factor=2.0 to extend to 8192 tokens but do not fine-tune. You test on a 6000-token prompt and observe that the model produces coherent SQL but achieves 12 pp lower accuracy than on 3000-token prompts. Propose two hypotheses for the accuracy drop.

**Q7.** Flash Attention 2 is required for long-context inference. Explain why in terms of memory complexity, and state the memory saving at 8192 tokens compared to standard attention.

## Deep Scenario

**Q8.** A customer comes to you with a PostgreSQL database that has 45 tables, 15 of which are TimescaleDB hypertables. Their largest schema is 7,200 tokens. They want to use your postgres-sqlcoder-7b for natural language queries across any of these tables.

Write a 200-word technical solution that: (a) evaluates whether context extension or schema compression (or both) is the right approach for this customer, (b) describes the implementation plan with estimated compute cost, and (c) identifies the residual risk after your proposed solution.
