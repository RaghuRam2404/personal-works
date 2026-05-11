# Week 65 Quiz — Local Deployment

## Multiple Choice

**Q1.** You build llama.cpp with `-DLLAMA_METAL=ON` and run with `--n-gpu-layers 35` but inference is still the same speed as CPU-only. What is the most likely cause?

A. Qwen2.5 models are not supported by the Metal backend
B. The GGUF file was created without Metal metadata; you must re-quantize
C. Ollama intercepts the Metal backend; you must use llama-server instead
D. The `cmake -B build` step used a cached configuration without Metal; you must delete the build directory and reconfigure

**Q2.** Your sql_ask.py prompt template differs from the training format by one line: it uses `## SQL Query` instead of `### SQL Query`. You notice accuracy drops from 81% to 67% on your benchmark. This drop is best explained by:

A. The model has memorized `###` as a literal token pattern that triggers SQL generation mode
B. `##` has a different tokenization to `###` so the model sees a different number of tokens
C. The model learned a strong association between the `### SQL Query` prefix and SQL output during SFT; deviating from it shifts the probability distribution away from valid SQL starts
D. The HTTP payload encoding treats `#` characters differently than `#`

**Q3.** A user reports that your CLI tool sometimes outputs the SQL followed by a paragraph of English explanation. What is the most likely Modelfile fix?

A. Decrease temperature from 0.1 to 0.0
B. Add the explanation start tokens to the `stop` list in the Modelfile (e.g., add `stop "This query"`)
C. Add the end-of-turn token `<|im_end|>` to the stop list and verify Ollama is using it
D. Increase `num_ctx` to prevent context truncation

**Q4.** You measure TTFT of 2.3 seconds for a 900-token schema prompt. Your user finds this too slow for interactive use. Which intervention will most directly reduce TTFT?

A. Increase `temperature` to 0.5 to let the model generate faster
B. Reduce the schema prompt length by providing only the relevant tables for each query
C. Switch from streaming to non-streaming mode
D. Increase `num_ctx` to accommodate longer schemas

## Short Answer

**Q5.** Explain why `temperature=0.0` may not produce greedy decoding in Ollama and what value you should use instead.

**Q6.** You want to deploy `sql_ask.py` so that a non-Python teammate can use it from their terminal without installing Python dependencies. Describe two approaches, each with one tradeoff.

**Q7.** The Modelfile `SYSTEM` prompt says "output only SQL." Your model still occasionally outputs Markdown code fences (```sql ... ```). Without retraining, propose two inference-time interventions to suppress the code fences.

## Deep Scenario

**Q8.** Your team wants to add a PostgreSQL execution step to the CLI tool: generate SQL, run it against a staging database, and show results. A colleague proposes adding `--execute` to `sql_ask.py` that pipes output to `psql`. You are responsible for the safety review.

Write a safety analysis (150–200 words) that: (a) identifies the top two risk categories from executing model-generated SQL, (b) proposes a specific technical mitigation for each, and (c) recommends one operational policy for the staging environment.
