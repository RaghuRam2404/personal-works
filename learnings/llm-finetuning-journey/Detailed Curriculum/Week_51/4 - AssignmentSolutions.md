# Week 51 Assignment Solutions

## Task 2 — Final Evaluation Sweep (Reference Script)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reward_fn import execute_sql, extract_sql

model_paths = {
    "v1": "<your-handle>/postgres-sqlcoder-7b-v1",
    "v2": "<your-handle>/postgres-sqlcoder-7b-v2-dpo",
    "v3": "<your-handle>/postgres-sqlcoder-7b-v3-grpo",
    "v3-iter1": "<your-handle>/postgres-sqlcoder-7b-v3-iter1",
    "v3-iter2": "<your-handle>/postgres-sqlcoder-7b-v3-iter2",
}

results = {}
for name, path in model_paths.items():
    print(f"Evaluating {name}...")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model.eval().cuda()
    
    exec_correct, sem_correct, syntax_errors, lengths = [], [], [], []
    
    for prompt, reference_sql in test_set:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        
        completion = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:])
        sql = extract_sql(completion)
        lengths.append(len(output[0]) - inputs["input_ids"].shape[1])
        
        if sql is None:
            exec_correct.append(False); sem_correct.append(False)
            syntax_errors.append(True); continue
        
        result = execute_sql(sql, db_dsn)
        ref_result = execute_sql(reference_sql, db_dsn)
        
        executes = result["success"] and result["row_count"] > 0
        exec_correct.append(executes)
        sem_correct.append(executes and set(map(tuple, result["rows"])) == 
                           set(map(tuple, ref_result["rows"])))
        syntax_errors.append("syntax" in (result.get("error") or "").lower())
    
    results[name] = {
        "exec_acc": sum(exec_correct) / len(exec_correct),
        "sem_acc": sum(sem_correct) / len(sem_correct),
        "syntax_err": sum(syntax_errors) / len(syntax_errors),
        "mean_len": sum(lengths) / len(lengths),
    }
    del model  # Free VRAM before loading next model
```

**Expected output table (representative, not actual):**

| Model | Exec Acc | Sem Acc | Syntax Err | Complex Exec Acc | Mean Length |
|---|---|---|---|---|---|
| v1 (SFT) | 68% | 44% | 22% | 40% | 118 tok |
| v2 (DPO) | 79% | 57% | 7% | 48% | 112 tok |
| v3 (GRPO) | 81% | 55% | 5% | 49% | 148 tok |
| v3-iter1 | 84% | 56% | 4% | 61% | 151 tok |
| v3-iter2 | 84% | 60% | 4% | 62% | 155 tok |

---

## Task 3 — Model Selection Framework (Filled Example)

| Criterion | v3-iter1 | v3-iter2 | Winner |
|---|---|---|---|
| Execution accuracy | 84% | 84% | Tie |
| Semantic accuracy | 56% | 60% | v3-iter2 |
| Complex exec acc | 61% | 62% | v3-iter2 |
| KL divergence | 3.2 nats | 4.1 nats | v3-iter1 |
| Mean gen length | 151 tok | 155 tok | v3-iter1 |

**Decision:** v3-iter2. The +4pp semantic accuracy improvement outweighs the slightly higher KL (4.1 vs 3.2 nats — both are within healthy range). The 4-token generation length increase is negligible for practical use. Semantic accuracy is the better signal for "does the model actually answer correctly" — a prerequisite for Phase 6's goal of beating GPT-4 on domain SQL.

---

## Task 4 — Phase 5 Final Report Key Points

The report should clearly state:

1. **Net improvement over SFT baseline:** v3 (best) achieves 84% exec acc vs 68% for v1 — a 16pp improvement through three stages of alignment.

2. **Each stage's contribution:**
   - SFT: established the schema-aware SQL foundation
   - DPO: reduced syntax errors by 15pp; did not help complex queries
   - GRPO: improved complex query handling by 13pp; slight semantic accuracy regression fixed by iteration

3. **Biggest lesson:** Reward quality (having reference SQL for semantic verification) matters more than reward quantity (more training steps). The semantic accuracy gap between v3 and v3-iter2 came from adding reference SQL to the reward function, not from more steps.

4. **Phase 6 roadmap:** Scale the dataset to 50K examples, specifically adding more complex join/CTE/TimescaleDB examples. Run the full SFT→DPO→GRPO pipeline fresh with the larger dataset.

---

## How to Verify You Did It Right

1. Final eval table has 5 models × 5 metrics with all cells filled.
2. `postgres-sqlcoder-7b-phase5-best` is accessible on HF Hub.
3. The best model beats v1 by ≥ 5pp on execution accuracy (Gate requirement is v3 > v1).
4. Phase 5 final report is ≤ 600 words but covers all 5 required sections.
5. You have a documented iteration log for every experiment run in Weeks 50–51.
