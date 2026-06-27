---
name: honest-methodology-feedback
description: "User wants methodological flaws surfaced proactively, not impressive-looking numbers"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 11e41294-9753-4062-90ba-19e8ece7c1f2
---

On Exp 1A Phase 3, I flagged that the headline LSTM/Transformer "win" was an artifact of a memoryless evaluator rather than real sequence-memory value, instead of presenting it as a clean win. The user immediately acted on it (chose Option A to fix the design).

**Why:** This is a research project aimed at a defensible finding/paper. A flawed metric that produces flattering numbers is worse than useless. The user explicitly instructed "do not tune to chase the holdout number… report honestly," and rewarded the honest flag with a design pivot.

**How to apply:** Before presenting any model "win," sanity-check whether the metric actually measures the intended construct (e.g., does the evaluator credit what the model is designed to exploit?). Surface confounds, ceilings, and circularity up front. Prefer an honest "this doesn't show what we wanted" over a tidy table. See [[option-a-terminal-reward]].
