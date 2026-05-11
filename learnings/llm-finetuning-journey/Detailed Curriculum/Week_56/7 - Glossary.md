# Week 56 Glossary

**CoSQL:** Conversational Text-to-SQL dataset (Yu et al., 2019) with 30K+ annotated dialogue turns across Spider databases, where users react to intermediate SQL results.

**SParC:** Sequential Paraphrase Corpus (Yu et al., 2019); multi-turn SQL dataset where users express sequential intents as progressive question refinements without seeing intermediate results.

**Multi-turn conversation:** A training or inference interaction with more than one user/assistant exchange; requires the model to maintain context across turns.

**Conversational context:** The accumulated history of prior turns in a multi-turn dialogue, including previously established tables, filters, and result schemas.

**Loss masking:** Setting the loss weight to zero for input tokens (system prompt, user turns) so the model only learns to predict assistant (SQL) tokens.

**DataCollatorForCompletionOnlyLM:** A TRL utility class that automatically masks all non-assistant tokens in a chat-format dataset during training.

**Response template:** The string delimiter (e.g., `<|im_start|>assistant\n`) used by the completion-only loss collator to identify where assistant responses begin.

**Context hallucination:** A failure mode where a model generates SQL referencing tables or columns not established in the prior conversational context.

**Implicit reference:** A user utterance that refers to an entity from a prior turn without naming it explicitly (e.g., "those sensors," "same period," "do the same for building B").

**Conversation coherence:** The property that each turn in a multi-turn conversation logically follows from and references the previous turn.

**sqlglot transpile:** The sqlglot library function for converting SQL between dialects (e.g., SQLite to PostgreSQL); essential for reusing CoSQL examples in a PostgreSQL context.

**pgloader:** A tool for migrating SQLite databases to PostgreSQL; useful for loading Spider SQLite databases into Postgres for CoSQL conversion validation.
