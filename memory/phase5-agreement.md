---
name: phase5-agreement
description: "Phase 5 Row-1 agreement analysis — BM reward-agnostic (100% agree); 18.6% disagreement is sparse-cell noise, REVISES row1-summary"
metadata: 
  node_type: memory
  type: project
  originSessionId: 11e41294-9753-4062-90ba-19e8ece7c1f2
---

Phase 5 — Row 1 cross-experiment agreement analysis (Bayesian Markov only; LSTM/Transformer excluded as degenerate). 107 holdout states, 3,263 pitches, γ=0.9.

**Reward-agnostic check (Analysis 5) — CONFIRMED:**
Bayesian Markov fits on transition frequencies and never sees the reward, so its recommendation is IDENTICAL across 1A/1B/1C for all 107 holdout states. Agreement map: 100% FULL AGREE, 0 partial, 0 disagree. Pairwise BM disagreement (1A-1B, 1A-1C, 1B-1C): all 0%.

**High-confidence agreement reference (Analysis 3) — the trustworthy Row-1 teaching output:**
29 states qualify (BM CI width < 0.20 AND ≥20 holdout pitches). Predominantly "throw 4-seam FB"; exceptions recommend slider (e.g. 0-2 & 1-2 vs RHH after FF; 2-1 vs RHH after FF) or changeup (2-1 vs LHH after FF). Most frequent: 0-0 vs RHH → FF (47%, n=434); 0-0 vs LHH → FF (52%, n=384). This reference is reward-independent (same for whiff/weak-contact/combined).

**High-confidence BM disagreement cards (Analysis 4): NONE** — BM is reward-agnostic.

**Supplementary reward-aware oracle (NOT BM) — the source of the old 18.6% figure:**
Oracle = argmax_a mean training G_t per (prev,count,hand,action) under each reward. Oracle pairwise disagreement (holdout-weighted): A-vs-B 18.6%, A-vs-C 15.8%, B-vs-C 2.9%.

**KEY REVISION (supersedes the "genuine coachable signal" claim in [[row1-summary]]):**
The 18.6% A-vs-B disagreement is SPARSE-CELL NOISE, not strategy. Its dominant case — 0-0 vs LHH, n=384 (63% of all disagreement pitches) — is whiff→SI (sinker) built on just **2 training pitches** vs weak-contact→FC (24). After requiring each reward's pick to rest on ≥10 training pitches AND ≥20 holdout pitches, **ZERO** reward-differentiated coaching cards survive. There is no trustworthy reward-aware pitch-type recommendation on this action space.

**Honest read:** Pitch-type-only gives a trustworthy reward-INDEPENDENT default (the 29-state FF-heavy reference) but CANNOT resolve the whiff-vs-weak-contact strategic question — the only reward-aware signal is too sparse to trust. Strongest motivation yet for Row 2 (location), where denser per-objective cells may make the tradeoff real and coachable.

Script: experiments/phase5_agreement.py. See [[row1-summary]], [[exp1a-finding]], [[project-status]].
