# Week 41 Resources

## Required Reading

- [Spinning Up in Deep RL — Introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — OpenAI / Josh Achiam. The clearest short introduction to MDPs, policies, and value functions. Read fully.
- [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — Lilian Weng's blog. Covers REINFORCE, actor-critic, PPO, and more. This week: focus on REINFORCE and baseline subtraction sections.

## Videos

- [HuggingFace Deep RL Course — Unit 1: Introduction to Deep RL](https://www.youtube.com/watch?v=vQgY9rf6uYQ) — HuggingFace — ~45 min. The YouTube companion to the HF Deep RL Course; watch Units 1, 2, and 4 in sequence this week to cover MDPs, Q-functions, and the REINFORCE algorithm.
- [David Silver — RL Course Lecture 7: Policy Gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs) — David Silver (DeepMind) — ~1h15m. The canonical academic treatment of REINFORCE and the policy gradient theorem; covers the math behind this week's algorithm.
- [Yannic Kilcher — REINFORCE and Policy Gradients Explained](https://www.youtube.com/watch?v=L_QZnKt21Oc) — Yannic Kilcher — ~30 min. Clear derivation of the REINFORCE update rule and why baseline subtraction reduces variance; pair with the Sutton 1999 paper.
- [Andrej Karpathy — Deep Reinforcement Learning: Pong from Pixels](https://www.youtube.com/watch?v=tqrcjHuNdmQ) — Andrej Karpathy — ~15 min. Intuitive motivation for policy gradient methods using a pixel-input game; good entry point before the CartPole implementation.

## Required Watching (HuggingFace Deep RL Course)

- [Unit 1 — Introduction to Deep RL](https://huggingface.co/learn/deep-rl-course/unit1/introduction) — HuggingFace. Covers RL framework, MDPs, value-based and policy-based approaches.
- [Unit 2 — Introduction to Q-Learning](https://huggingface.co/learn/deep-rl-course/unit2/introduction) — HuggingFace. Value functions and Bellman equations. Read Unit 2 for the Q-function and value function math; skip the implementation sections (you are not building DQN).
- [Unit 4 — Policy Gradient with PyTorch](https://huggingface.co/learn/deep-rl-course/unit4/introduction) — HuggingFace. Directly relevant — covers REINFORCE in PyTorch. This is your coding reference.
- [Unit 8 — PPO](https://huggingface.co/learn/deep-rl-course/unit8/introduction) — HuggingFace. Preview of next week; skim the intro section only this week.

## Papers

- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) — Sutton et al. 1999. The original policy gradient theorem paper. Read the theorem statement and proof (2 pages).

## Blog Posts

- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — Shengyi Costa Huang. Extremely practical. Read after Week 42, but bookmark now.
- [RL from Human Feedback: From Zero to chatGPT](https://huyenchip.com/2023/05/02/rl-human-feedback-rlhf.html) — Chip Huyen. High-level overview of how RL connects to RLHF. Good orientation for where Phase 5 is heading.

## GitHub Repos

- [CleanRL](https://github.com/vwxyzjn/cleanrl) — Single-file, research-quality RL implementations in PyTorch. The `ppo.py` file is the gold standard for a clean PPO. Study the REINFORCE logic in `reinforce.py` for comparison.
- [HuggingFace Deep RL Course notebooks](https://github.com/huggingface/deep-rl-class) — Official notebooks for the course units. Unit 4 notebook has the CartPole REINFORCE starter code.

## Documentation

- [PyTorch torch.distributions](https://pytorch.org/docs/stable/distributions.html) — Reference for `Categorical`, `Normal`, and other distributions you will use for sampling and log-prob computation.
- [Gymnasium CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) — Environment spec: obs space, action space, reward structure, termination conditions.

## Optional / Bonus

- [Reinforcement Learning: An Introduction (Sutton & Barto)](http://incompleteideas.net/book/the-book-2nd.html) — Free online. Chapter 13 covers policy gradient methods rigorously. This is the textbook definition; use it to verify your derivation.
- [Spinning Up PPO docs](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — The GAE section is required reading for Week 42. Skim it now.
