# Week 71 Assignment Solutions

## Task 1: Tulu 3 — Key Extractions

What to look for in the RLVR section:

RLHF uses a learned reward model (a separate neural network trained on human preference data) to score model outputs. This introduces a second optimization target that can be wrong or over-optimized. RLVR (and your GRPO) uses a verifiable reward — a function that objectively checks correctness (code execution, math equality, SQL result set). No learned reward model is needed; the ground truth check is the reward.

Your GRPO training is essentially RLVR with a custom reward function: `r(y) = 1.0 if execute(y) == execute(y_gold)`. Tulu 3 validates this at larger scale and broader task coverage.

Three reusable insights from Tulu 3:
- On-policy data generation: use your current checkpoint to generate SQL, verify executability, add successful examples to the next training run. Expected improvement: 1–3 pp based on Tulu 3's on-policy vs off-policy ablation.
- Preference data quality filter: re-examine your 5K DPO pairs using the chosen/rejected gap criterion (pairs where chosen score >> rejected score are higher signal). Filter to the top 2K pairs and see if DPO quality improves.
- SFT with decreasing temperature: generate SFT examples with temperature 1.0 first (for diversity), then fine-tune on the verifiably-correct subset. Tulu 3 reports 1–2 pp improvement from this curriculum.

## Task 2: SmolLM2 — Tokenizer Check

To count tokens for a SQL expression in Qwen2.5's tokenizer:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

expr = "time_bucket(ts, INTERVAL '1 hour')"
tokens = tok.encode(expr)
print(f"{len(tokens)} tokens: {tok.convert_ids_to_tokens(tokens)}")
```

Typical result: `time_bucket` → 2 tokens (`time`, `_bucket`); full expression → 8–10 tokens. This is inefficient for SQL generation. The practical implication: for your Week 75 base model comparison, check each candidate model's tokenization of common SQL keywords — a model with better SQL token coverage will have shorter effective sequence lengths and faster inference.

## Task 3: OLMo 2 — What Intermediate Checkpoints Enable

One example: OLMo 2 releases checkpoints at 20%, 40%, 60%, 80%, and 100% of pretraining. Researchers could track when capabilities like multi-step reasoning emerge during training — they found that arithmetic capabilities emerge sharply around 40% of training, not gradually. This phase transition would be invisible with only the final checkpoint. For your ablation study, this validates saving intermediate checkpoints (CPT-only, SFT-only) even if it costs storage, because the intermediate results tell you when each capability emerged.

## Task 4: Synthesis Template

```markdown
## What all three papers agree on

1. Data quality beats data quantity: all three filter aggressively before training.
2. Verifiable rewards (code execution, SQL execution) outperform learned reward models.
3. Staged training (pretraining → SFT → alignment) with increasing domain specificity
   consistently outperforms one-stage approaches.
4. On-policy data generation (using the current model to generate training data) 
   improves alignment quality.
5. Evaluation benchmarks diverge from real-world use; domain-specific evals matter.

## Three techniques to apply in Weeks 75–77

1. On-policy SQL generation (Tulu 3): after Week 75's base model comparison,
   use the best model to generate 5K new SQL examples, verify, and add to SFT data.
2. Two-phase CPT with quality upweighting (OLMo 2): for the bilingual Tamil model
   (Week 77), run a second CPT phase with Tamil SQL examples upweighted 3x.
3. SmolLM2 tokenizer audit (Week 75): check SQL keyword tokenization for all
   candidate base models; prefer the one with best SQL token coverage.
```

## Common Gotchas

- Tulu 3's RLVR results are on general instruction following; the SQL-specific transfer may be smaller — do not assume the same absolute gain.
- SmolLM2's data quality results apply at 1.7B parameters; effects at 7B may differ.
- OLMo 2's mid-training data mixing is done at pretraining scale (trillions of tokens); your CPT is much smaller. The mechanism is the same but the magnitude is different.

## How to Verify You Did It Right

Your synthesis is successful if you can answer: "What is one specific change I will make to my training pipeline based on these papers?" If the answer is vague ("improve data quality"), the reading was not deep enough. If the answer is specific ("use on-policy SQL generation after Week 75 SFT, verify executability, add 2K passing examples to Week 76 training"), the reading transferred into actionable knowledge.
