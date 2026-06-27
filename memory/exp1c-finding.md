---
name: exp1c-finding
description: "Exp 1C (pitch type / Reward C combined) results + weight sensitivity — type-only can't resolve reward objectives"
metadata: 
  node_type: memory
  type: project
  originSessionId: 11e41294-9753-4062-90ba-19e8ece7c1f2
---

Experiment 1C — Pitch Type / Reward C (Combined)
Weights: whiff = 1.0, weak contact = 0.7 (default, logged)
Holdout: 2023, γ = 0.9, terminal reward only

Results (default weights):
  Random Policy       E[G_t] = 0.711  SE = 0.012
  Empirical Frequency E[G_t] = 0.735  SE = 0.007
  Markov Chain        E[G_t] = 0.736  SE = 0.007
  Bayesian Markov     E[G_t] = 0.736  SE = 0.007  ← platform model
  LSTM                E[G_t] = 0.814  SE = 0.010  ⚠️ degenerate
  Transformer         E[G_t] = 0.832  SE = 0.012  ⚠️ degenerate

NN robustness:
  LSTM seed std = 0.013, Transformer seed std = 0.026
  Transformer dominant pitch: KC→SL→FF→FC→KC across seeds
  Transformer policy: 49% KC / 42% FC — off-distribution
  LSTM fallback rate: 5.0% (at threshold)
  Transformer fallback rate: 9.4% (exceeds threshold)

Weight sensitivity:
  Default (1.0/0.7): Bayesian Markov E[G_t] = 0.736, TVD = 0.629
  Alt (0.8/0.9):     Bayesian Markov E[G_t] = 0.706, TVD = 0.625
  Alt weights marginally closer to Cole's real tendencies but
  TVD ~0.63 in both cases — reward-max oracle wants ~46% cutters
  regardless. Weight tuning does not fix the action space problem.
  Recommendation: alt weights (0.8/0.9) noted but default (1.0/0.7)
  retained for consistency across experiments.

Key difference vs 1A and 1B:
  Bayesian Markov pitch recommendations identical across 1A/1B/1C.
  Reward function choice does not change type-only recommendations —
  confirms that pitch-type action space lacks sufficient resolution
  to distinguish reward objectives.

See [[exp1a-finding]], [[exp1b-finding]], [[row1-summary]].
