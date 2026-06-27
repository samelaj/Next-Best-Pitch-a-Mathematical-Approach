---
name: project-status
description: "Current state of Next Best Pitch — phase 4 complete, locked evaluator/baselines, next session pickup"
metadata: 
  node_type: memory
  type: project
  originSessionId: 11e41294-9753-4062-90ba-19e8ece7c1f2
---

Project: Next Best Pitch — A Mathematical Approach
Pitcher: Gerrit Cole (2021-2023)
Last updated: Phase 4 complete

CURRENT STATUS: Phase 4 complete. Ready for Phase 5
(Row 1 cross-experiment agreement analysis) pending confirmation.

Completed phases:
  Phase 1 — Environment and data setup          ✅
  Phase 2 — Exp 1A baselines                    ✅
  Phase 3 — Exp 1A primary models               ✅
  Phase 4 — Exp 1B and 1C primary models        ✅
  README and architecture diagrams              ✅

Pending:
  Phase 5 — Row 1 agreement analysis            🔲
  Row 2 — Location only (Exps 2A/2B/2C)         🔲
  Row 3 — Type + location (Exps 3A/3B/3C)       🔲
  Secondary models (GRU, MLP, BLR, HMM)         🔲
  Per-pitch shaping sensitivity run             🔲
  Cross-row comparison                          🔲
  Platform integration design                   🔲

Locked evaluator:
  γ = 0.9, MC return, terminal reward only
  Baselines (2023 holdout):
    Random Policy       E[G_t] = 0.4754  SE = 0.0072
    Empirical Frequency E[G_t] = 0.4958  SE = 0.0046
    Markov Chain        E[G_t] = 0.4949  SE = 0.0046
    Bayesian Markov     E[G_t] = 0.4949  SE = 0.0046

Locked terminal reward table:
  Strikeout            +1.5
  Out on contact       +0.8
  Single               -0.9
  Walk / HBP           -1.0
  XBH                  -1.3
  HR                   -1.6

Row 1 platform model: Bayesian Markov
NN status: degenerate on pitch-type-only — fixes queued for Rows 2-3
Key teaching insight: 18.6% state disagreement between whiff-optimal
and weak-contact-optimal under Bayesian Markov

Next session pickup:
  Read this file first.
  Then read memory/row1-summary.md.
  Then read CLAUDE.md.
  Current task: Phase 5 — Row 1 agreement analysis.
  Prompt: map all Bayesian Markov disagreement states across
  1A vs 1B, 1A vs 1C, and 1B vs 1C. Produce coaching-readable
  output for each disagreement state.
