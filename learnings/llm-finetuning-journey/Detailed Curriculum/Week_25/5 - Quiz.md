# Week 25 Quiz — Dataset Construction

## Multiple Choice

**Q1.** When fine-tuning Qwen2.5-Coder-7B, you must format your data in ChatML format matching the model's chat_template. During training, the loss should be computed:

A) On all tokens in the conversation, including system and user turns  
B) Only on the assistant's response tokens (the SQL query)  
C) Only on the first 50 tokens of the assistant response  
D) On the system prompt tokens only  

---

**Q2.** Self-Instruct (Wang et al. 2022) generates synthetic training data by:

A) Mining instruction-response pairs from the Common Crawl web corpus  
B) Using a language model to generate new instructions from seed examples, then generating responses  
C) Human annotators paraphrasing existing instruction datasets  
D) Training a separate instruction-generation model on GPT-4 outputs  

---

**Q3.** You have 500 hand-written PostgreSQL examples and 5,000 GPT-generated synthetic examples. For fine-tuning quality, you should:

A) Discard the hand-written examples — they are too few to influence training  
B) Up-weight the hand-written examples by repeating them more frequently in the training mix  
C) Use only the synthetic examples to avoid bias from hand-written styles  
D) Hand-written examples should only be used for validation, never training  

---

**Q4.** Your Spider-to-PostgreSQL conversion must filter out `GROUP_CONCAT` because:

A) It is a function that computes vector dot products and is unrelated to SQL  
B) It is a SQLite-specific aggregate function; PostgreSQL uses `STRING_AGG` instead  
C) It causes infinite loops in PostgreSQL query planning  
D) `GROUP_CONCAT` is deprecated in all SQL dialects since 2022  

---

**Q5.** When including schema information in your training prompts, the most important reason to include `CREATE TABLE` statements is:

A) To help the model learn DDL syntax by reading many CREATE TABLE examples  
B) To give the model the table names, column names, and data types needed to write correct SQL  
C) To teach the model to recognize when schemas are poorly designed  
D) PostgreSQL requires CREATE TABLE before every query at runtime  

---

## Short Answer

**Q6.** Explain in 3–4 sentences why "the model must only see the SQL query in the assistant turn" is important — i.e., why you should strip explanatory text from assistant responses in your training data.

---

**Q7.** You are converting 1,500 Spider examples to ChatML format. Spider uses SQLite-compatible SQL. List 4 specific SQL constructs that are valid in SQLite but invalid or different in PostgreSQL, and give the PostgreSQL equivalent for each.

---

**Q8.** Your self-instruct generation produces 3,000 instructions, but after running your quality filter, only 1,800 pass. You need 2,900 for your v1 dataset. List 3 options for getting the remaining 1,100 examples without lowering your quality bar.

---

## Scenario

**Q9.** You are building the v1 dataset for your PostgreSQL/TimescaleDB SQL assistant. Your colleague suggests: "Let's just download the entire WikiSQL dataset (80K examples) and use that as our fine-tuning data."

1. What are 2 structural problems with WikiSQL for your specific task (PostgreSQL/TimescaleDB)?
2. WikiSQL was created in 2017. What SQL features does it lack that are essential for your TimescaleDB use case?
3. If you did use WikiSQL, what processing steps would be needed to make it useful?
4. Given your three-tier plan (Spider/BIRD + hand-written + self-instruct), explain why 5,000 high-quality examples likely outperforms 80,000 WikiSQL examples for your deployment scenario.
