# Week 28 Quiz — Fine-Tuning Fundamentals

## Multiple Choice

**Q1.** You have a pretrained `Qwen2.5-Coder-7B` model. Your downstream task is PostgreSQL text-to-SQL. You have 8K labeled (question, schema, SQL) pairs and 50GB of raw PostgreSQL documentation. Which training strategy is most likely to give the best final performance given your compute budget of ~$50 on Colab Pro?

A. Continued pretraining on the 50GB documentation, then nothing else  
B. Continued pretraining on 50GB docs, then SFT on 8K pairs  
C. SFT on 8K pairs directly, skipping continued pretraining  
D. Only run DPO on the 8K pairs, treating them as preference data

---

**Q2.** In the InstructGPT SFT stage, the loss is computed over which tokens?

A. All tokens in the prompt + response concatenated  
B. Only the prompt tokens  
C. Only the response tokens  
D. A randomly sampled 50% of all tokens

---

**Q3.** A colleague says: "SFT teaches the model new factual knowledge about our company's internal systems." Which response is most accurate?

A. Correct — SFT updates all weights so all knowledge can change  
B. Partially correct — SFT can teach new vocabulary but not new factual associations  
C. Incorrect — SFT primarily shapes output format and activates existing knowledge; it cannot reliably inject new facts absent from pretraining  
D. Incorrect — only continued pretraining can teach new knowledge; SFT changes nothing structural

---

**Q4.** The intrinsic low-rank hypothesis (the theoretical basis of LoRA) states:

A. Fine-tuning loss landscapes are always convex  
B. The weight changes during fine-tuning lie in a low-dimensional subspace of the full parameter space  
C. Only the top-k attention heads need to be updated during fine-tuning  
D. The rank of the full weight matrix decreases during fine-tuning

---

**Q5.** In the modern post-training pipeline (SFT → DPO → GRPO), which stage is best suited to improve SQL execution correctness using binary pass/fail signals?

A. SFT — because it directly trains on correct SQL  
B. DPO — because it ranks correct SQL above incorrect SQL  
C. GRPO — because it uses verifiable execution-based scalar rewards  
D. Continued pretraining — because it exposes the model to more SQL text

---

## Short Answer

**Q6.** You are advising a team building a customer support chatbot. They have: (a) 500GB of support ticket raw text (unlabeled), (b) 50K (customer question, ideal response) pairs. They plan to do only SFT on the 50K pairs and skip continued pretraining. Under what condition is this a correct decision? Under what condition would you add continued pretraining first?

---

**Q7.** In InstructGPT, the SFT stage used only ~13K examples, yet the model dramatically improved over the base GPT-3. How is this possible given that GPT-3 was trained on 300B tokens? What does this tell you about the relationship between pretraining and fine-tuning data efficiency?

---

**Q8.** Describe the risk of catastrophic forgetting in SFT. Give one concrete symptom you would observe in a text-to-SQL model that has catastrophically forgotten pretrained knowledge. Then give one practical mitigation.

---

## Scenario

**Q9.** You are fine-tuning `Qwen2.5-Coder-7B` on 5K PostgreSQL text-to-SQL pairs. After 3 epochs, the model generates perfect SQL for all training examples but produces incoherent garbage on the held-out test set. The model also refuses to answer general coding questions it answered correctly before fine-tuning.

Provide a diagnosis with at least 3 hypotheses ranked by likelihood, and recommend specific changes to your training setup.
