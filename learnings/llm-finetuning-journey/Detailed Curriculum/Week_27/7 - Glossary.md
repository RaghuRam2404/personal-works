# Week 27 Glossary — Phase 3 Gate

Terms specific to the gate process and Phase 3 retrospective. Prior-week terms are not repeated.

---

**Gate criterion**: A specific, evidence-backed requirement that must be met before progressing to the next phase of the curriculum.

**PASS**: Gate status meaning all evidence is present and meets the minimum quantitative bar.

**CONDITIONAL**: Gate status meaning evidence is partial but sufficient to proceed, with a commitment to complete remediation within the first two weeks of Phase 4.

**FAIL**: Gate status meaning a core deliverable is entirely missing; requires one additional remediation week before Phase 4 begins.

**Remediation**: Targeted work to close a specific skill or artifact gap identified during a gate assessment.

**Phase 3 retrospective**: A written self-reflection (1–2 pages) documenting what you learned, what took longer than expected, what you would do differently, and what is now solid.

**Evidence-based self-certification**: Assessing your own readiness using concrete, verifiable artifacts (links, file paths, code outputs) rather than subjective confidence.

**postgres-sql-v1**: The name for your v1 PostgreSQL Text-to-SQL dataset, stored on HuggingFace Hub, with ≥5,000 ChatML-formatted examples and ≥98% SQL validity.

**Phase 4**: The next phase of the curriculum (Weeks 28–40) covering the full fine-tuning stack: SFT, LoRA, and QLoRA applied to Qwen2.5-Coder-7B on your domain dataset.

**Compute budget accounting**: The practice of tracking total GPU cost incurred during a training phase against the hard ceiling ($200 total for this course), to ensure remaining phases have adequate headroom.

**MFU (Model FLOP Utilization)**: The fraction of theoretical peak GPU FLOP/s actually achieved during training; used to convert wall-clock time into compute cost. Reference value: 35% on A100.
