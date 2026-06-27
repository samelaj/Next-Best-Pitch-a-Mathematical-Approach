---
name: exp1a-finding
description: "Exp 1A result under Option A terminal-reward MDP — NNs don't earn complexity; Bayesian Markov wins"
metadata: 
  node_type: memory
  type: project
  originSessionId: 11e41294-9753-4062-90ba-19e8ece7c1f2
---

**Exp 1A (pitch type / terminal-reward MDP, γ=0.9, Reward A framing), 2023 holdout, expected G_t:**
Random 0.475 · Markov 0.495 · Bayesian Markov 0.495 (==Markov, α=1) · Empirical 0.496 · Transformer 0.504 · LSTM 0.524 (seed-0).

**Honest conclusion: the LSTM/Transformer do NOT earn their complexity on Row 1.**
- Frequency baselines barely beat Random (+0.02, ~2–3 SE): **pitch TYPE alone has weak leverage on plate-appearance outcomes** (location/execution/sequencing matter more).
- NN "edge" (+0.03–0.06 over Markov) is **within the random-init noise band**: across seeds 0–4, LSTM 0.526±0.022, Transformer 0.555±0.027, and the dominant recommended pitch flips between FC/FF/SL/KC/CH per seed. Not reproducible.
- NN policies are **degenerate & off-distribution** (e.g. LSTM ~42% FC, Transformer ~74% KC; Cole throws FF ~90%). Baselines sensibly recommend FF ~90%.
- NN fallback rate >5% (LSTM 5.5%, Transformer 7.9%) — they exploit sparse/optimistic Q* cells for rare pitches. Baselines <0.5%.
- Bayesian Markov: stable, interpretable, ties the empirical optimum, gives calibrated uncertainty (11% of states flagged CI width >0.3).

**Platform implication:** build the Row-1 teaching tool on **Bayesian Markov**, not the NNs.

**Recommended architectural changes (not numeric tuning) before NNs could help:** richer action space (location, Rows 2–3); add back the deferred per-pitch shaping to densify signal; behavior-regularized / conservative-Q to keep policies near the pitcher's real distribution; optional league base-model pretraining then per-pitcher fine-tune. See [[option-a-terminal-reward]], [[project-status]], [[honest-methodology-feedback]].
