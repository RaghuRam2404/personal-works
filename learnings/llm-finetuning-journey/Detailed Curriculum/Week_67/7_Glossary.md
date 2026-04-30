# Week 67 Glossary — Technical Report Writing

**Technical report**: A structured document describing a system's design, training, evaluation, and limitations; prioritizes reproducibility over theoretical novelty.

**Abstract**: A self-contained 150–200 word summary covering problem, method, results, and release; should be readable independently of the paper.

**Contribution**: A deliverable your work adds to the field (dataset, model, code, benchmark); distinct from a finding (measurement).

**Finding**: An empirical observation from your experiments (accuracy number, ablation result); distinct from a contribution (deliverable).

**Data contamination**: Overlap between training data and evaluation data that artificially inflates measured accuracy; must be explicitly ruled out.

**Reproducibility check**: Verification that a third party can re-run your experiments from the published code and data and obtain numbers within a stated tolerance.

**Related work section**: Paper section that situates your contribution relative to prior work, organized by research cluster rather than chronologically.

**Author-year citation**: Academic citation style (e.g., "Yu et al. 2018") used in ML papers instead of numbered references.

**Ablation study**: An experiment that removes one component of your system to measure its isolated contribution; essential for justifying design choices.

**Preprint**: A paper posted to a public server (arXiv, OpenReview) before peer review; used in ML to establish priority and gather community feedback.
