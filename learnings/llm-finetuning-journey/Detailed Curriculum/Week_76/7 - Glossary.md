# Week 76 Glossary — Multi-Turn Agentic SQL with Tool Use

**Agentic loop:** An inference-time control structure that repeatedly calls a model, executes its tool calls, and feeds results back until a stopping condition.

**Tool call:** A structured JSON emission by the model specifying a function name and arguments to be executed by the orchestrating runtime.

**Tool-result turn:** A conversation turn containing the output of a tool execution, used as feedback to the model in the next generation step.

**Multi-turn SFT:** Supervised fine-tuning on conversations with more than one user-assistant exchange, requiring per-turn loss masking.

**Correction rate:** Fraction of initially incorrect SQL queries that the agentic model successfully fixes in subsequent rounds.

**First-attempt EM:** Exact match accuracy of the model's first SQL emission before any correction round; measures single-shot baseline preservation.

**Loss masking:** Setting label values to -100 for tokens on which no gradient should be computed (user turns, tool-result turns in multi-turn SFT).

**ReAct (Reason + Act):** An agentic pattern where the model generates natural-language reasoning before each tool call to guide its action.

**Synthetic error injection:** A data augmentation technique that corrupts correct SQL with rule-based mutations to generate first-round failures for trajectory training.

**Silent failure:** A query execution that returns rows without errors but produces semantically incorrect results; not detectable from error messages alone.

**Orchestrator:** The Python runtime layer that intercepts model tool calls, executes them against real systems, and manages the conversation loop.

**Over-refinement:** An agentic failure mode where the model continues issuing tool calls after reaching a correct answer, creating unnecessary latency and correctness risk.
