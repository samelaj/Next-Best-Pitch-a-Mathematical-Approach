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

CURRENT STATUS: Phase 5 complete. Ready for Row 2 (location-only
action space) pending confirmation.

Completed phases:
  Phase 1 — Environment and data setup          ✅
  Phase 2 — Exp 1A baselines                    ✅
  Phase 3 — Exp 1A primary models               ✅
  Phase 4 — Exp 1B and 1C primary models        ✅
  Phase 5 — Row 1 agreement analysis            ✅
  README and architecture diagrams              ✅

Pending:
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
Phase 5 result (REVISES old 18.6% insight, see [[phase5-agreement]]):
  Bayesian Markov is reward-agnostic → 100% agreement across 1A/1B/1C.
  Trustworthy output = 29 high-confidence reward-INDEPENDENT states
  (mostly "throw 4-seam FB"). The 18.6% whiff-vs-weak-contact
  "disagreement" was the reward-aware oracle and is SPARSE-CELL NOISE
  (top case = sinker on 2 training pitches); ZERO reliable
  reward-differentiated cards survive. Type-only cannot resolve the
  strategic tradeoff → motivates Row 2.

Row 2 architectural changes for NNs (from row1-summary.md):
  1. Richer action space (location 9-zone, then type+location)
  2. Per-pitch shaping rewards to densify the learning signal
  3. Conservative-Q / behavior regularization (keep policy near
     the pitcher's real pitch distribution)
  4. League base-model pretraining → per-pitcher fine-tune
  Reminder: run the per-pitch shaping SENSITIVITY run after Row 2.

Next session pickup:
  Read this file first.
  Then read memory/row1-summary.md and memory/phase5-agreement.md.
  Then read CLAUDE.md.
  Current task: Row 2 — location-only action space (Exps 2A/2B/2C).
  Primary question: does location sequencing add signal that
  type-only could not resolve? Secondary: do the NNs stabilize on
  the richer action space? Follow the same phase structure
  (baselines → primary → agreement), stop at phase boundaries.
