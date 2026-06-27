---
name: exp1b-finding
description: "Exp 1B (pitch type / Reward B weak contact) results — NNs degenerate, Bayesian Markov platform"
metadata: 
  node_type: memory
  type: project
  originSessionId: 11e41294-9753-4062-90ba-19e8ece7c1f2
---

Experiment 1B — Pitch Type / Reward B (Weak Contact)
Holdout: 2023, γ = 0.9, terminal reward only

Results:
  Random Policy       E[G_t] = 0.337  SE = 0.0065
  Empirical Frequency E[G_t] = 0.341  SE = 0.0036
  Markov Chain        E[G_t] = 0.345  SE = 0.0035
  Bayesian Markov     E[G_t] = 0.345  SE = 0.0035  ← platform model
  LSTM                E[G_t] = 0.408  SE = 0.0060  ⚠️ degenerate
  Transformer         E[G_t] = 0.462  SE = 0.0059  ⚠️ degenerate

NN robustness:
  LSTM seed std = 0.011 (3x eval SE), dominant pitch flips across seeds
  Transformer seed std = 0.016 (4x eval SE), dominant pitch flips
  LSTM policy: 51% cutters vs Cole actual 7.2% — off-distribution
  Transformer policy: 44% cutters — off-distribution
  LSTM fallback rate: 6.9% (exceeds 5% threshold)
  Transformer fallback rate: 9.6% (exceeds 5% threshold)

Bayesian Markov:
  Mirrors Cole's real pitch mix (89% FF)
  11% of states flagged high-uncertainty (CI width > 0.3)
  Identical point policy to Markov Chain — value is calibrated
  uncertainty, not a better point estimate

Launch data note:
  12 terminal balls-in-play missing exit velocity
  Treated as hard contact (+0.4) per documented assumption

Key difference vs 1A:
  Bayesian Markov E[G_t] shifts from 0.495 (1A) to 0.345 (1B)
  reflecting the different reward scale — not a performance change.
  Pitch recommendations are identical across 1A and 1B under
  Bayesian Markov. Reward function does not change what to throw
  on a type-only action space.

See [[exp1a-finding]], [[exp1c-finding]], [[row1-summary]].
