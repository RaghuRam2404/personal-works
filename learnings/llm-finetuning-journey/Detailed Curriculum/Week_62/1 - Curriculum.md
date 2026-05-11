# Week 62 — Comprehensive Eval Part 2: Head-to-Head vs Frontier Models

## Learning Objectives

By the end of this week, you will be able to:

- Run a fair, reproducible head-to-head comparison between your model and GPT-4o, Claude 3.5, SQLCoder, and DeepSeek-Coder-V2
- Apply identical evaluation protocols across models to ensure comparability
- Analyze where your model wins and loses relative to each competitor
- Write a head-to-head evaluation section suitable for a technical report
- Compute per-model costs (inference cost per query × accuracy) to make the efficiency argument

## The Head-to-Head Comparison

This week's output is the central table of your technical report — the one that answers: "Does your specialized 7B model beat frontier generalist models on your domain?" A credible answer requires identical prompts, identical databases, identical result comparison logic, and honest reporting of where you lose as well as where you win.

## Concepts

### Models to Compare

**Your model:** `postgres-sqlcoder-7b-final` (7B parameters, locally deployable)

**Baselines (same family):**
- `Qwen2.5-Coder-7B` — the base model you started from. Measures: how much did your training help?
- `SQLCoder-7B` (Defog) — the most prominent specialized SQL model. Direct competitor.

**Frontier closed models:**
- `GPT-4o` (OpenAI) — the current state of the art for general SQL tasks. Your ceiling comparison.
- `Claude 3.5 Sonnet` (Anthropic) — strong coding capability, good comparison point.

**Open frontier model:**
- `DeepSeek-Coder-V2-Lite` (16B) — the open model most likely to approach GPT-4o performance. Important because it is your size class at 16B (vs your 7B).

### Ensuring Fair Comparison

**Identical prompts:** Every model receives the same schema DDL and the same natural language question. The only difference is the system prompt's framing — some models need slightly different instructions (e.g., Claude prefers explicit "Return only SQL, no explanation" instruction). Document any prompt differences.

**Temperature:** All models run at temperature=0 for deterministic, reproducible evaluation.

**Schema format:** Same DDL format for all models. Do not add column comments only for your model.

**API models:** GPT-4o and Claude require API calls. Budget: ~$10–15 for 200 examples × 2 models × $0.02–0.05 per call. Use a batching approach and cache responses to avoid re-paying for repeated calls.

**Result comparison:** Identical result comparison logic (sorted rows, rounded floats) for all models.

### The Domain Advantage Hypothesis

Your central hypothesis: a specialized 7B model will outperform generalist models on TimescaleDB-specific queries, even if it underperforms on general SQL benchmarks. Test this hypothesis explicitly by breaking down results by:

1. Standard PostgreSQL queries (WHERE, GROUP BY, JOIN) — generalists may win here
2. Advanced PostgreSQL features (window functions, CTEs, lateral joins) — competitive
3. TimescaleDB-specific queries (time_bucket, hyperfunctions, continuous aggregates) — you should win here

If your hypothesis is confirmed (you win on category 3 even if losing on 1 and 2), this is a compelling and honest result that demonstrates the value of domain specialization.

### Cost-Adjusted Performance

A key argument for using your model in production: cost efficiency. Compute:

```
Cost-per-correct-query = (inference cost per query) / (execution accuracy)
```

For your model running locally (Ollama, Mac): cost ≈ $0 (electricity only)
For GPT-4o: ~$0.02–0.05 per query
For Claude 3.5: ~$0.01–0.03 per query

At 70% accuracy:
- Your model: $0 / 0.70 = $0 per correct query (hardware amortized)
- GPT-4o: $0.03 / 0.83 = $0.036 per correct query

Even if GPT-4o is 13% more accurate, if you need 10,000 correct queries per month, GPT-4o costs $360 vs. $0 for your local model. For a small startup or individual developer, this is the decisive argument.

### The "Hard Cases" Analysis

Beyond overall accuracy, analyze the specific questions where you beat GPT-4o and where you lose:

**Where you might beat GPT-4o:**
- TimescaleDB hyperfunctions with precise syntax (GPT-4o has seen fewer of these)
- Queries involving your specific schema patterns (you trained on them; GPT-4o hasn't)
- Continuous aggregate syntax (rare in GPT-4o's pretraining data)

**Where GPT-4o likely beats you:**
- Novel schema structures not in your training data
- Complex multi-step reasoning about which tables to join
- Queries requiring external domain knowledge not expressed in the schema

Document 5 examples from each category in your technical report. These qualitative examples are often more persuasive than the headline accuracy numbers.

### Statistical Validity of the Comparison

With 200 custom benchmark examples, your 95% CI is approximately ±7 percentage points. This means: if your model scores 76% and GPT-4o scores 83%, the 7-point difference is within 2× the CI margin — not statistically significant. You need ~500+ examples for 95% CI ≈ ±4.5pp, which would make a 7pp difference statistically significant.

Action: run your evaluation on all 200 examples, report CI via bootstrapping, and be honest about statistical significance.

### Common Misconceptions and Pitfalls

**"I should compare on BIRD to maximize comparability."** True for academic credibility. But your central claim is about the PostgreSQL/TimescaleDB domain — this is most honestly measured on your custom benchmark, even if it is not a standard academic benchmark. Include both.

**"If I beat GPT-4o on one metric, I've won."** Nuanced result reporting: clearly state which metrics and subsets you compare on. Cherry-picking is immediately visible to any critical reader.

**"SQLCoder is not a fair comparison — it's also specialized."** It is the most relevant specialized competitor. If you beat SQLCoder on your domain, that is a strong result.

## Time Allocation (6–8 hrs)

- 1h: Set up API clients for GPT-4o and Claude; test on 5 examples
- 1h: Run base Qwen2.5-Coder-7B evaluation (same harness, different model path)
- 1h: Run SQLCoder-7B evaluation
- 1h: Run GPT-4o evaluation (200 custom + 100 BIRD — budget: ~$15)
- 1h: Run Claude 3.5 evaluation (200 custom + 100 BIRD — budget: ~$8)
- 0.5h: Run DeepSeek-Coder-V2-Lite (local or via API)
- 1.5h: Compile comprehensive comparison table; per-model error analysis; 5 wins/losses examples
- 0.5h: Write `head_to_head_comparison.md`; commit

## Connections

This week builds directly on Week 61, which set up the BIRD-SQL and Spider 2.0 evaluation harnesses and ran your first systematic benchmark. Week 61 produced the infrastructure (evaluation scripts, result comparison logic, bootstrap CI code); Week 62 uses that same harness to extend the comparison to frontier closed models. The Phase 4 evaluation work (Weeks 36–38, establishing your domain benchmark and baseline) is the other essential predecessor: without your Custom-200 benchmark and the baseline execution accuracy numbers from Phase 4, you have no domain-specific comparison surface that frontier models can be tested against.

Weeks 63–64 use the results you produce here to make quantization tradeoffs. Specifically, the per-category breakdown from this week (standard PostgreSQL vs. advanced PostgreSQL vs. TimescaleDB-specific) informs which query types are most sensitive — and which are therefore most at risk from quantization precision loss. If Week 62 reveals that your model's advantage concentrates in TimescaleDB hyperfunctions, Week 63's quantization study should prioritize measuring accuracy degradation on exactly those query types. The head-to-head comparison table produced this week also becomes a central table in the technical report you write in Weeks 67–70.
