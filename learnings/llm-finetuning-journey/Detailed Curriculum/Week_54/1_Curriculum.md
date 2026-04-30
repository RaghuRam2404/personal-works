# Week 54 — Synthetic Data Generation: Magpie and Genie Approaches

## Learning Objectives

By the end of this week, you will be able to:

- Explain the Magpie self-instruct approach and how it differs from classic Self-Instruct
- Explain the Genie approach to generating grounded, schema-aware synthetic data
- Design teacher model prompts that produce high-quality, diverse PostgreSQL/TimescaleDB SQL pairs
- Build an async generation pipeline that calls the teacher API, validates execution, and saves results
- Target your generation budget toward under-represented skills identified in Week 53's gap analysis

## Concepts

### The Synthetic Data Landscape

Before the modern era of strong teacher models, synthetic data was primarily generated via templates — manually crafted rules that substituted values into fixed SQL patterns. This produced large but shallow datasets. The revolution came when models like GPT-4 became strong enough to serve as teachers: you describe a task in natural language, and the teacher generates diverse, novel examples.

Two families of approach dominate in 2024–2025:

**Self-Instruct (Wang et al., 2023):** Start with a seed set of human-written examples. Use the LLM to generate new tasks by prompting: "Here are N tasks. Generate M new tasks that are diverse and different from these." Then generate solutions for the new tasks. The key insight is that LLMs can generate richer instruction sets than humans can write by hand, at scale.

**Magpie (Xu et al., 2024, arXiv 2406.08464):** A cleverer approach that exploits the chat template directly. Instead of explicit task generation, Magpie feeds only the system prompt + the beginning of the user turn to the model, and lets the model complete both the instruction AND the response. This means the model generates its own training data in a self-consistent style, with no external oracle needed. The resulting data is on-policy for that model family.

**Genie (Yehudai et al., 2024, arXiv 2401.14367):** Focuses on grounded generation — generating examples conditioned on specific schemas, documents, or contexts. For text-to-SQL, Genie-style generation means you provide a database schema (DDL) and ask the teacher to generate question/SQL pairs that are specifically grounded in that schema. This solves the hallucinated-column-name problem endemic to unconstrained SQL generation.

### Magpie for SQL: The Self-Instruct Path

Magpie applied to SQL generation looks like this:

```
System: You are an expert PostgreSQL and TimescaleDB engineer. When given a database schema, you write precise, efficient, idiomatic SQL queries.

[User turn prefix — stopped here, model generates the rest]
```

The teacher model completes the user question based on the system prompt context, then generates the SQL answer. The advantage: the generated examples are coherent (the question matches the answer style) and stylistically on-policy for that teacher model family.

For your use case, you augment Magpie with schema injection: include a DDL in the system prompt so the model generates questions that are grounded in a real schema.

### Genie for SQL: The Grounded Path

Genie-style generation is more controlled. You write an explicit generation prompt:

```
Given the following PostgreSQL schema:
{schema_ddl}

Generate {n} text-to-SQL pairs where:
- The natural language question is realistic and domain-appropriate
- The SQL answer is valid PostgreSQL and executable
- Include at least one query using {target_skill}
- Difficulty: {difficulty_level}

Format: Return a JSON list of {"question": ..., "sql": ...} objects.
```

This approach gives you precise skill targeting, which is critical for filling the gaps identified in Week 53. The trade-off: the teacher may produce repetitive phrasings or subtly hallucinate column names.

### Prompt Engineering for High-Quality SQL Generation

The quality of your synthetic data is entirely determined by your teacher prompts. Several lessons from the literature:

**Explicit difficulty control.** "Generate a hard query" produces medium-difficulty output. "Generate a query that requires a lateral join with a correlated subquery in the WHERE clause" produces hard output. Be specific about the SQL constructs required.

**Schema grounding.** Always include the full DDL, including data types, constraints, and indexes. The teacher needs to know which columns are timestamptz to correctly generate time-series queries.

**Few-shot examples in the prompt.** Include 2–3 gold-standard examples (from your hand-curated set) in each teacher prompt. This dramatically improves output quality for niche skills the teacher has seen rarely.

**Output format specification.** Tell the teacher to return JSON. Parse the output with a try/except. If parsing fails, discard the response. Aim for a parse rate > 90%; if below, simplify your output format spec.

**Diversity pressure.** Add to each prompt: "The query must use different column names and table combinations than these examples: {recent_examples}." Without this, the teacher tends to generate paraphrases of its most recent outputs.

### Building the Generation Pipeline

Your pipeline must handle:

1. **Async API calls** — you are generating 30K examples; sequential calls would take days. Use `asyncio` + `httpx` or the OpenAI async client.
2. **Execution validation** — after each batch, run SQL against Postgres. Track execution rate per skill.
3. **Deduplication on the fly** — add each generated example's MinHash to your LSH index before saving.
4. **Resumability** — if the pipeline crashes at example 15,000, you must be able to restart from 15,000, not 0. Save a checkpoint every 100 examples.
5. **Rate limiting** — OpenAI and Anthropic APIs have rate limits. Implement exponential backoff.

### Cost Estimation

At GPT-4o pricing (~$5/M input tokens, ~$15/M output tokens as of 2024):
- Each generation prompt: ~1,500 tokens input, ~500 tokens output
- 30K examples / 5 examples per API call = 6,000 API calls
- Input: 6,000 × 1,500 = 9M tokens = ~$45
- Output: 6,000 × 2,500 = 15M tokens = ~$225

This exceeds budget. Optimizations:
- Generate 5–10 examples per API call (amortizes input tokens)
- Use Claude Haiku or GPT-4o-mini for easy examples (~10× cheaper)
- Reserve GPT-4o for Expert-difficulty examples only
- Target 15K synthetic examples + combine with 5K augmented from existing data

Budget-realistic plan: ~$30–50 outside the $200 course ceiling, or ~15K examples with a mix of teacher strengths.

### Common Misconceptions and Pitfalls

**"The teacher model is always right."** GPT-4o generates incorrect SQL for niche constructs it has seen rarely (TimescaleDB hyperfunctions, `EXCLUDE` in window frames). Your execution filter in Week 55 is not optional — it is the quality gate.

**"I can generate all 30K examples in one weekend."** Rate limits, parsing failures, and execution validation mean 30K examples realistically takes 3–5 days of pipeline runtime. Start the pipeline on Monday.

**"Few-shot examples in the prompt guarantee quality."** They help but don't guarantee. The teacher may ignore them for skills it is over-confident about. Always validate.

## Connections

This week directly consumes Week 53's gap analysis (which skills to target) and data card (which schemas to use). Week 55 will filter this week's output. The generation prompts you write this week become the reproducibility appendix of your technical report (Week 67–70).

## Time Allocation (6–8 hrs)

- 1h: Read Magpie paper (arXiv 2406.08464) — focus on Section 3 (methodology)
- 1h: Read Genie paper (arXiv 2401.14367) — focus on Section 3 (grounded generation)
- 2h: Write and test teacher prompts (start with 10 examples, verify execution)
- 2.5h: Build the async generation pipeline; run first 1,000 examples
- 0.5h: Log metrics to W&B; commit everything
