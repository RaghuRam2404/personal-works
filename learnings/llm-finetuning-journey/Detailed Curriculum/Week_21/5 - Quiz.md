# Week 21 Quiz — Running Pretraining

## Multiple Choice

**Q1.** Your 56M model is training at 50,000 tokens/sec on an A100. Targeting 2B tokens, approximately how many wall-clock hours will the training take?

A) 4 hours  
B) 11 hours  
C) 22 hours  
D) 40 hours  

---

**Q2.** At step 5,000 your train loss spikes from 3.8 to 8.2. After 200 steps it recovers to 3.9. What is the most appropriate response?

A) Stop training immediately and restart with half the learning rate  
B) Log the spike in your training journal and continue training without changes  
C) Reduce the batch size and resume from the checkpoint before the spike  
D) Increase the gradient clip to 5.0 to prevent future spikes  

---

**Q3.** You are using bf16 mixed precision. Compared to fp16, bf16:

A) Has higher numerical precision (23-bit mantissa vs 10-bit)  
B) Has a wider dynamic range (8-bit exponent vs 5-bit) making overflow less likely  
C) Requires a GradScaler to prevent underflow  
D) Is slower on A100 because A100 is optimized for fp16  

---

**Q4.** Your gradient norm is consistently being clipped to 1.0 (every step fires the clip). This most likely indicates:

A) The model is working correctly — gradient clipping every step is expected  
B) The learning rate is too high relative to the current loss landscape  
C) The dataset contains corrupted tokens that inflate gradients  
D) bf16 is causing numerical overflow in the backward pass  

---

**Q5.** After a Colab disconnection, you resume from a checkpoint at step 15,000. Your DataLoader resets to the beginning of `train.bin`. What is the consequence?

A) Training will fail because data was seen in the wrong order  
B) The model will train on some data for a third time (since it already saw it in epoch 1 and part of epoch 2), adding one more fractional epoch  
C) The model will exhibit catastrophic forgetting of the first 15,000 steps  
D) The optimizer states become inconsistent with the gradient history  

---

## Short Answer

**Q6.** Your 56M model's val loss is 3.2 after 30,000 steps. Compute the perplexity. What does this number mean intuitively?

---

**Q7.** You see the following W&B training curve: loss drops from 10.4 to 4.0 in the first 500 steps, then plateaus around 4.0 for 10,000 more steps with no improvement. List 3 hypotheses for why the loss has plateaued, and for each, describe one diagnostic check.

---

**Q8.** A colleague says you should set `grad_clip=10.0` to "let the gradients flow freely" during early training. Explain why this is wrong and what gradient clipping actually does.

---

## Scenario

**Q9.** Your model is at step 8,000 (approx 500M tokens) with val loss = 3.9. Suddenly, your Colab session disconnects. You saved a checkpoint at step 6,000.

1. You resume from step 6,000. How many tokens did you "waste" by having to redo steps 6,000–8,000?
2. Your `train.bin` has 800M tokens total. After the resume, is there a risk of the model seeing the same tokens more than twice? Explain.
3. You notice the optimizer states in the checkpoint were not saved correctly (only model weights were saved). What problem does this cause for the resumed training?
4. Design a simple checkpoint saving strategy that would have prevented this issue, using only 2 checkpoint files (not one per step).
