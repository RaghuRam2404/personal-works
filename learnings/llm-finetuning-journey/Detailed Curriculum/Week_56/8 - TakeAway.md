# Week 56 TakeAway — Multi-Turn SQL Data

**One-liner:** Multi-turn training requires correct loss masking — loss only on assistant turns, all of them.

---

## Chat Template Format

```python
# Build messages list for SFTTrainer
messages = [
    {"role": "system", "content": f"Expert PostgreSQL engineer.\nSchema:\n{ddl}"},
    {"role": "user", "content": turn1_q},
    {"role": "assistant", "content": turn1_sql},
    {"role": "user", "content": turn2_q},
    {"role": "assistant", "content": turn2_sql},
]
# Loss computed ONLY on assistant content tokens
```

## Loss Masking with TRL

```python
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# Mark assistant response delimiter
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)
# Collator automatically masks everything before each assistant turn
```

## CoSQL SQLite → PostgreSQL Conversion

```python
import sqlglot
pg_sql = sqlglot.transpile(sqlite_sql, read="sqlite", write="postgres")[0]
# Watch for: strftime → EXTRACT, GROUP_CONCAT → string_agg
```

---

## Decision Rules

- Compute loss on ALL assistant turns in a conversation (not just the last)
- Mask system prompt and user turns to loss weight 0
- If conversation > 4,096 tokens: keep last 3 turns; do not use automatic truncation mid-query
- Target 20–25% of training examples as multi-turn
- Verify coherence: turn N+1 must reference at least one entity from turn N
- TimescaleDB multi-turn should include gap-filling as a common turn 3 or 4 refinement

---

## Numbers to Remember

- CoSQL: 30K+ turns, 200 Spider databases (mostly not PostgreSQL-native)
- SParC: ~4K conversations, 2–5 turns
- Expected valid PostgreSQL examples from CoSQL: ~2,000–3,000
- Target multi-turn count in v3: ≥ 4,000 conversations
- Max recommended conversation length: 4 turns (4,096 token budget)
- p90 token length target for full dataset: ≤ 2,048 tokens

---

## Red Flags

- Loss computed on user tokens: collator misconfigured, remove immediately
- Zero multi-turn conversations in the dataset: format conversion bug
- All multi-turn from CoSQL only, no TimescaleDB-specific: model will not generalize to your domain
- Conversation turn 2 has 0% entity overlap with turn 1: synthetic generation is incoherent
