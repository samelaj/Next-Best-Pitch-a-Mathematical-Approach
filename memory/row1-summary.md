---
name: row1-summary
description: "Row 1 (pitch-type-only) complete — cross-experiment summary, 18.6% disagreement, NN fixes queued for Rows 2-3"
metadata: 
  node_type: memory
  type: project
  originSessionId: 11e41294-9753-4062-90ba-19e8ece7c1f2
---

Row 1 Summary — Pitch Type Only Action Space
Experiments: 1A (Whiff), 1B (Weak Contact), 1C (Combined)
Status: COMPLETE

Cross-experiment results table:
  Model            1A      1B      1C
  Random           0.475   0.337   0.711
  Empirical        0.496   0.341   0.735
  Markov           0.495   0.345   0.736
  Bayesian Markov  0.495   0.345   0.736
  LSTM             0.524   0.408   0.814
  Transformer      0.504   0.462   0.832
(Scales differ across A/B/C — compare within column only)

Row 1 confirmed finding:
  Bayesian Markov is the platform model for pitch-type-only.
  NN raw scores are artifacts of seed lottery, degenerate
  off-distribution policies, and fallback exploitation — not
  coachable strategy. The cutter anomaly (NNs collapsing to
  FC at 42-51% vs Cole's 4.5-7.2% actual rate) is the
  mechanism: sparse evaluator cells price the cutter
  generously because few observations exist to penalize
  bad recommendations in those states.

Bayesian Markov is reward-agnostic on type-only:
  Pitch recommendations identical across 1A, 1B, and 1C.
  Only the E[G_t] scale changes. Reward function choice
  does not resolve to different pitch type recommendations
  on this action space.

Key teaching insight (18.6% disagreement states):
  Whiff-optimal and weak-contact-optimal strategies disagree
  on 18.6% of holdout-weighted states under Bayesian Markov.
  Top disagreement: 0-0 vs LHH (n=384 pitches)
    Whiff-optimal recommends: SI (sinker)
    Weak-contact-optimal recommends: FC (cutter)
  This is a genuine coachable signal — what you throw depends
  on what you are trying to induce. Phase 5 agreement analysis
  will map all disagreement states.

  ** REVISED by Phase 5 (see [[phase5-agreement]]): the 18.6% is
  the reward-aware ORACLE's disagreement, NOT Bayesian Markov's
  (BM is reward-agnostic → 100% agreement). And the 18.6% is
  SPARSE-CELL NOISE: the dominant 0-0 LHH case (63% of it) is
  whiff->SI built on just 2 training pitches. After a ≥10-train /
  ≥20-holdout reliability bar, ZERO reward-differentiated cards
  survive. Type-only cannot resolve the whiff-vs-weak-contact
  tradeoff — motivates Row 2. The trustworthy teaching output is
  the 29 high-confidence reward-INDEPENDENT agreement states. **

NN architectural fixes queued for Rows 2-3:
  1. Richer action space (location, type+location)
  2. Per-pitch shaping rewards to densify learning signal
  3. Conservative-Q / behavior regularization to prevent
     policy drift from real pitch distribution
  4. League base model pretraining → per-pitcher fine-tune

Row 2 queued: location-only action space
  Primary question: does location sequencing add signal
  that type-only cannot resolve?
  Secondary question: do NNs stabilize on a richer
  continuous action space?

See [[exp1a-finding]], [[exp1b-finding]], [[exp1c-finding]], [[project-status]].
