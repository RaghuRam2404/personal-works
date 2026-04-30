# Week 78 Assignment — Final Phase 6 Gate and Course Wrap

## Setup Checklist

- [ ] HuggingFace account with write access to your namespace (`<user-handle>`)
- [ ] `huggingface_hub` library installed: `pip install huggingface_hub`
- [ ] GitHub account; repository for your training code ready (can be new or existing)
- [ ] arXiv account (register at arxiv.org; you will need an institutional endorsement OR another arXiv author to endorse you for cs.CL — plan this in advance; alternatively, use HuggingFace papers as fallback)
- [ ] Your final model checkpoint at `postgres-sqlcoder-7b-final`
- [ ] Your quantized model files (GGUF Q4_K_M, Q5_K_M from Week 63–64)
- [ ] Your technical report draft (from Weeks 67–70)
- [ ] W&B project `week-78-gate` created for logging final evaluation run

---

## Task 1 — Capstone Deliverables Audit

**Goal:** Produce a complete, honest inventory of what exists and what is missing.

**Requirements:**
- [ ] Create `week78/capstone_audit.md` with a table listing every deliverable from the checklist in Curriculum.md.
- [ ] For each deliverable, mark status: Done / In Progress / Missing.
- [ ] For each Missing item, write one sentence explaining why it is missing and whether you will complete it now or defer it with a specific date.
- [ ] Priority: any deliverable marked "Missing" that would prevent someone from replicating your results must be completed this week.

**Format:**
```markdown
| Deliverable | Status | Notes |
|---|---|---|
| postgres-sqlcoder-7b-final on HF Hub | Done | https://huggingface.co/<handle>/... |
| GGUF Q4_K_M quantization | Done | ... |
| Technical report PDF | In Progress | Submitting to arXiv this week |
```

**Deliverable:** `week78/capstone_audit.md`.

---

## Task 2 — Final Model Card and Hub Upload

**Goal:** Produce a complete, externally legible model card and upload all artifacts.

**Requirements:**
- [ ] Write `week78/model_card.md` using the format below. This becomes the README for your HuggingFace model repo.
- [ ] Required sections: Model Summary, Benchmark Results (table format), Training Recipe, Usage (copy-paste code snippet), Limitations, License.
- [ ] Upload the model card and confirm the HuggingFace repo is public.
- [ ] Verify the usage snippet in your model card actually runs end-to-end (test it locally before publishing).

**Benchmark table (minimum required):**

| Benchmark | Metric | postgres-sqlcoder-7b | GPT-4o | Claude 3.5 Sonnet |
|---|---|---|---|---|
| Custom-200 (TimescaleDB) | EM | 83.1% | 79.4% | 81.2% |
| BIRD-SQL dev | EX | 68.4% | — | — |
| Spider 1.0 | EM | 82.7% | — | — |

**Deliverable:** `week78/model_card.md` + confirmation link to public HuggingFace repo.

**Hints:** Use `huggingface_hub.HfApi().upload_folder(folder_path="...", repo_id="<handle>/postgres-sqlcoder-7b-final", repo_type="model")` for bulk upload.

---

## Task 3 — Final Evaluation Run and Results Log

**Goal:** Run one final, clean evaluation on Custom-200 with your published checkpoint and log it as the official capstone result.

**Requirements:**
- [ ] Load the model from HuggingFace Hub (not your local checkpoint — this verifies the upload worked).
- [ ] Run Custom-200 evaluation (EM computation, same script as your earlier evaluation weeks).
- [ ] Log results to W&B project `week-78-gate` with run name `final-capstone-eval`.
- [ ] Save results to `week78/final_eval_results.json` with structure: `{"custom200_em": ..., "model": "...", "checkpoint": "...", "date": "..."}`
- [ ] If the result deviates from 83.1% by more than 0.5%, investigate before publishing.

**Deliverable:** `week78/run_final_eval.py` + `week78/final_eval_results.json` + W&B run link.

---

## Task 4 — Blog Post and Retrospective

**Goal:** Make your work legible to the outside world and to yourself.

**Requirements (Blog Post):**
- [ ] Write a 500–800 word blog post at `week78/blog_post.md` covering: the problem you set out to solve, what you built, your key result, one thing that surprised you, and how someone can use or extend your model.
- [ ] Write for a technically informed audience (software engineers, data engineers) who are not ML researchers. Avoid jargon that is not explained.
- [ ] Publish it somewhere publicly accessible: Medium, Substack, your personal site, LinkedIn article, or HuggingFace blog. Paste the public URL in `week78/capstone_audit.md`.

**Requirements (Retrospective):**
- [ ] Write `week78/course_retrospective.md` covering at minimum: three things you learned that surprised you, two things that were harder than expected, one major decision point where you could have gone differently (and what you would choose now), and what you want to do in the next 6 months as a direct result of this course.
- [ ] Length: 400–700 words. This is for you, not for anyone else.

**Deliverable:** `week78/blog_post.md` + `week78/course_retrospective.md`.

---

## Stretch Goals

- Submit your technical report to arXiv under cs.CL. Getting an endorsement takes 1–2 weeks; start the process now. If you cannot get endorsement in time, publish on HuggingFace Papers instead.
- Record a 5-minute demo video: screen recording of the Ollama CLI generating SQL from a natural language query, with the Custom-200 benchmark numbers shown. Upload to YouTube or Loom and link from your model card.
- Open a GitHub Discussions thread on your code repository asking for feedback. Even one constructive comment from the community is validation that your work is externally legible.
- Apply the capstone model to one real query your current employer or a personal project actually needs. Document the result honestly: did it work out of the box, or did it need prompt adjustment?
