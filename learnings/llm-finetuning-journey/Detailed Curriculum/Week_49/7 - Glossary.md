# Week 49 Glossary

**KTO (Kahneman-Tversky Optimization)**: Alignment method based on prospect theory that works with unpaired (desirable/undesirable) annotations instead of paired preference examples.

**Prospect theory**: Behavioral economics model by Kahneman & Tversky showing humans are more sensitive to losses than equivalent gains; used as the mathematical basis for KTO.

**Desirable/undesirable**: KTO's label format: a completion is either desirable (positive) or undesirable (negative), replacing the DPO chosen/rejected pair format.

**ORPO (Monolithic Preference Optimization)**: Alignment method that combines SFT loss and preference learning in a single stage without a reference model, using an odds ratio penalty.

**Odds ratio (ORPO)**: The ratio of the probability that the model assigns to a completion vs. its complement: π(y|x) / (1 − π(y|x)); used as the reward signal in ORPO.

**SimPO (Simple Preference Optimization)**: Alignment method that uses average log-probability (normalized by sequence length) as the reward with a margin γ, requiring no reference model.

**Length normalization (SimPO)**: Dividing the cumulative log-probability by sequence length (1/|y| · log π(y|x)) to make the reward independent of response length.

**Margin γ (SimPO)**: A hyperparameter requiring the chosen response to be preferred by at least γ log-probability units over the rejected response.

**Alignment zoo**: Informal term for the collection of competing preference optimization methods (DPO, KTO, ORPO, SimPO, IPO, etc.); this week surveys the most practically important ones.

**Monolithic training**: A training approach that combines multiple objectives (e.g., SFT + preference) in a single training stage (ORPO) rather than sequential stages (SFT → DPO).

**IPO (Identity Preference Optimization)**: A DPO variant that avoids the Bradley-Terry assumption; mentioned in ORPO and SimPO papers as a comparison point.
