---
name: option-a-terminal-reward
description: Why/what of the Option A pivot to a terminal/delayed-reward MDP for pitch sequencing
metadata: 
  node_type: memory
  type: project
  originSessionId: 11e41294-9753-4062-90ba-19e8ece7c1f2
---

On 2026-06-27 the project switched from an **immediate per-pitch reward scored by a memoryless (count, handedness, action) Q-table** to **Option A: a terminal/delayed-reward MDP**.

**Why:** A Phase-3 diagnostic showed the per-pitch + memoryless-table design was structurally incapable of crediting sequence memory. Q depended only on (count, stand, action), so every policy was capped by a memoryless reward-max oracle (≈0.319), and the LSTM/Transformer "wins" over Markov (0.13 vs 0.06) only reflected reward-greedy action selection vs frequency-matching — NOT that deeper memory helps. The core research question (does sequence memory add value?) was untestable.

**What Option A does:** at-bat = episode; reward realized mainly at the terminal pitch (PA outcome from `events`) and credited back across the sequence as a discounted return `G_t = Σ_{k≥t} γ^(k−t) r_k`, γ default 0.9 (logged hyperparameter). The per-pitch A/B/C tables become per-pitch shaping rewards summed into the return; a shared terminal PA-outcome table (strikeout +1.5, out-on-contact +0.8, single −0.8, walk/HBP −1.0, XBH −1.2, HR −1.6 — initial/tunable) dominates. Evaluation becomes **history-aware, return-based**: models predict Q(history-state, a)=E[G_t]; tabular Markov/Bayesian-Markov condition on (prev pitch, count, hand) earning partial history credit. The old memoryless table is kept only as a deprecated reference ceiling.

**How to apply:** all 9 experiment cells now use terminal-return reward + history-aware evaluation. Always log γ and Reward-C weights. See [[project-status]].
