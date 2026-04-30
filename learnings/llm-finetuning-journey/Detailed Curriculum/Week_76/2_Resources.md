# Week 76 Resources — Multi-Turn Agentic SQL with Tool Use

## Papers

[ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629) — Foundational paper on the Reason + Act agentic pattern; directly applicable to SQL self-correction loop design.

[DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction (Pourreza & Rafiei, 2023)](https://arxiv.org/abs/2304.11015) — Demonstrates multi-step SQL generation with error self-correction; reports 3–5 pp EM gains over single-shot.

[DAIL-SQL: Efficient Prompt Engineering for Text-to-SQL (Gao et al., 2023)](https://arxiv.org/abs/2308.15363) — Systematic study of SQL prompting strategies including execution feedback; baseline comparison for your agentic results.

[CHESS: Contextual Harnessing for Efficient SQL Synthesis (Talaei et al., 2024)](https://arxiv.org/abs/2405.16755) — State-of-the-art agentic SQL system with multi-step correction and schema linking; useful for understanding the ceiling performance.

[Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)](https://arxiv.org/abs/2302.04761) — Covers self-supervised tool-use training; the approach to generating tool-call training data is relevant to your trajectory construction.

[FireActs: Toward Language Agent Fine-tuning (Chen et al., 2023)](https://arxiv.org/abs/2310.05915) — Fine-tuning smaller models on agentic trajectories; shows that SFT on agent trajectories outperforms using the loop with a non-agentic model.

## Videos

- [LangChain — Tool Calling and Agents Tutorial](https://www.youtube.com/watch?v=DjuXACWYkkU) — LangChain — ~35 min. Practical walkthrough of implementing tool-calling loops; useful for understanding the orchestrator pattern before building your own.
- [Andrej Karpathy — Software 2.0 and LLM Agents](https://www.youtube.com/watch?v=9EN_HoEk3KY) — Andrej Karpathy — ~20 min. Conceptual framing for why agentic loops are the right abstraction for complex task completion.

## Blog Posts / Articles

[Hugging Face — Tool Use with Transformers](https://huggingface.co/docs/transformers/en/chat_templating#tool-use--function-calling) — Official documentation on `apply_chat_template` with `tools=` argument; shows exact format for Llama, Mistral, and other architectures.

[Simon Willison — The ReAct Pattern for LLM Tools](https://simonwillison.net/2023/Apr/2/a-closer-look-at-the-react-pattern/) — Clear explanation of the loop mechanics; useful mental model before implementing your own.

[Modal — Running PostgreSQL in a Serverless Function](https://modal.com/docs/examples/postgres) — If you need a cloud-hosted PostgreSQL for your agentic loop instead of local.

## GitHub Repos

[langchain-ai/langchain — SQL Agent](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/agent_toolkits/sql/base.py) — Reference implementation of a SQL agent with tool use; study the loop logic, not the LangChain abstractions.

[huggingface/trl — SFTTrainer](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py) — Source for understanding how to extend `DataCollatorForCompletionOnlyLM` to multi-turn scenarios.

[vanna-ai/vanna](https://github.com/vanna-ai/vanna) — Open-source Text-to-SQL framework with execution feedback; useful for comparing your agentic approach against a production system.

## Documentation

[psycopg2 — Exception Classes](https://www.psycopg.org/docs/errors.html) — Complete list of PostgreSQL error codes as Python exceptions; use these to write targeted error-type detection in your orchestrator.

[OpenAI — Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) — Reference for the JSON tool-call schema format; even if you are not using OpenAI, the schema spec is widely adopted.

## Optional / Bonus

[Gorilla: Large Language Model Connected with Massive APIs (Patil et al., 2023)](https://arxiv.org/abs/2305.15334) — Training LLMs specifically for tool use with API documentation retrieval; extends the agentic concept beyond SQL.

[AgentBench: Evaluating LLMs as Agents (Liu et al., 2023)](https://arxiv.org/abs/2308.03688) — Benchmark for agentic task performance; useful for understanding where your SQL agent sits relative to general-purpose agents.
