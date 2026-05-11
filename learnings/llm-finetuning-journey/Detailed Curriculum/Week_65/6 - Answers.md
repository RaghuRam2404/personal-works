# Week 65 Answers

## Q1

**Answer: D**

**Why correct:** CMake caches build variables in `CMakeCache.txt` inside the build directory. If you previously configured without Metal and then add `-DLLAMA_METAL=ON`, the cached value overrides your new flag. Deleting the build directory forces a clean reconfiguration that picks up the Metal flag correctly. You can also use `cmake -B build -DLLAMA_METAL=ON --fresh` to force cache reset without deleting the directory.

**Why others are wrong:**
- A: llama.cpp has had Qwen2.5 GGUF support since mid-2024.
- B: GGUF files do not contain backend metadata; the backend is chosen at runtime by the host binary.
- C: Ollama uses llama.cpp internally and does use Metal; this is not an Ollama limitation.

---

## Q2

**Answer: C**

**Why correct:** During supervised fine-tuning, the model sees thousands of examples where `### SQL Query` is followed by the SQL answer. The model learns a strong conditional probability: given the sequence `### SQL Query\n`, the next token is very likely to be a SQL keyword (`SELECT`, `WITH`, `INSERT`). When you change the trigger to `## SQL Query`, the conditional probability distribution shifts — the model has seen far fewer examples of this prefix and is less confident that SQL should follow. The 14 percentage point drop is a real-world demonstration of the sensitivity of instruction-tuned models to format drift.

**Why others are wrong:**
- A: "Memorized as literal token pattern" is imprecise — it is a learned conditional distribution, not rule-based memorization.
- B: The tokenization difference exists but is not the primary mechanism; the issue is distributional, not structural.
- D: HTTP payload encoding has no effect on the prompt content seen by the model.

---

## Q3

**Answer: C**

**Why correct:** Qwen2.5 chat models use the ChatML format, where `<|im_end|>` marks the end of each turn. If this token is not in the Ollama stop list, the model generates the SQL (correct), then sees the implicit continuation of the conversation and starts generating an "assistant explanation" response. Adding `<|im_end|>` as a stop token causes Ollama to halt generation when the model signals end-of-turn, before the explanation begins.

**Why others are wrong:**
- A: `temperature=0.0` affects sampling stochasticity, not the stop condition; the model will still generate explanations just more deterministically.
- B: Adding specific English phrases as stop tokens is fragile and won't catch all explanation patterns.
- D: Increasing `num_ctx` addresses truncation, not post-SQL generation.

---

## Q4

**Answer: B**

**Why correct:** TTFT is dominated by the time to process the input prompt (the prefill step), which scales linearly with prompt token count. A 900-token schema is expensive to prefill. By providing only the subset of tables relevant to each query (e.g., 2–3 tables instead of 15), you can reduce the prompt to 200–300 tokens and drop TTFT from 2.3s to under 0.5s. This is the most impactful, practical intervention.

**Why others are wrong:**
- A: Temperature affects sampling during generation (decode phase), not prefill time.
- C: Streaming vs non-streaming does not change when the first token is computed; it only changes when it is delivered to the client.
- D: Increasing `num_ctx` adds overhead, not removes it.

---

## Q5

**Model answer:** Many inference engines, including Ollama and llama.cpp, treat `temperature=0.0` as a sentinel value meaning "use the server default" rather than mathematically setting temperature to zero (which would make softmax outputs degenerate). The mathematical reason is that softmax with temperature T produces probabilities proportional to exp(logit / T); as T → 0, this approaches argmax (greedy decoding), but at exactly T = 0 it causes division by zero. In practice, use `temperature=0.01` — close enough to greedy that results are effectively deterministic across runs while avoiding the sentinel/division-by-zero issue.

---

## Q6

**Model answer:** Option 1: Compile to a self-contained binary using PyInstaller (`pyinstaller --onefile sql_ask.py`). The resulting binary bundles the Python interpreter and all dependencies. Tradeoff: the binary is large (50–80 MB) and platform-specific — you need separate builds for macOS and Linux. Option 2: Wrap the tool as a Bash script that calls `curl` to hit the Ollama REST API directly, reading the schema file and question as arguments. Tradeoff: the Bash script cannot handle streaming, error parsing, or retry logic as cleanly as Python, and the JSON wrangling via `jq` becomes brittle for complex responses.

---

## Q7

**Model answer:** Two interventions: First, add a grammar constraint using llama.cpp's GBNF grammar feature (`--grammar-file sql.gbnf`) that restricts output tokens to valid SQL characters and keywords. This prevents backticks and markdown fences from ever being generated, as they are not in the SQL grammar. Second, apply a post-processing step in `sql_ask.py` that strips any ` ```sql ` or ` ``` ` fences from the output string using a regex: `re.sub(r"```(?:sql)?\n?", "", output)`. This is a quick defensive measure that does not require any model changes.

---

## Q8 — Deep Scenario

**Model answer:** Risk 1: Data mutation. Even on a staging database, the model may generate `DELETE`, `UPDATE`, `DROP`, or `TRUNCATE` statements in response to ambiguous questions. Mitigation: parse the generated SQL before execution using `sqlparse` or `pglast`; reject any statement whose top-level node is not `SELECT`. Add a strict allowlist: only `SELECT` statements execute; everything else prints the SQL for human review.

Risk 2: Injection via schema or question input. If the schema file or question string is sourced from an untrusted user, a carefully crafted input could steer the model toward generating SQL that exfiltrates or corrupts data. Mitigation: run the SQL through a database user with `SELECT`-only privileges on the staging schema. Even if the model generates a malicious `DELETE`, the database will reject it with a permissions error.

Operational policy: the staging database should be a read-only replica or a daily snapshot restore — never connected to any write path. Document this policy in the tool's README so future maintainers understand why the database credentials used are intentionally read-only.
