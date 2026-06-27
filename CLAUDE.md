# CLAUDE.md — Pitcher Sequencing Optimization

> Research project within the Pathway Performance platform.
> Owner: Jake Samela (JSamela-Dev)

---

## Project Purpose

Build a **prescriptive pitch sequencing optimization system** using MLB Statcast data. Given the current at-bat state and pitches thrown so far, the model recommends the next pitch (type, location, or both) to maximize a defined outcome objective.

This is not a descriptive fingerprinting system — it is an **optimal sequencing policy engine** framed as a Markov Decision Process (MDP).

End goal: a mathematically grounded teaching tool that tells a pitcher, after throwing a fastball in a specific count and location, what to throw next and where — and explains why based on historical outcome data.

---

## Core Framing

**Sequential decision problem (MDP):**
- **State** = (pitches thrown so far in at-bat, count, batter handedness, outcomes so far)
- **Action** = next pitch (type only | location only | type + location)
- **Reward** = **terminal at-bat (plate-appearance) outcome**, credited back across the pitch sequence as a discounted return `G_t = Σ_{k≥t} γ^(k−t) · r_k` (discount γ, default 0.9 — a logged hyperparameter). The per-pitch outcome tables (Reward Functions, below) act as **per-pitch shaping rewards** `r_k` summed into the return; the dominant term is the terminal PA outcome. **(Option A — adopted 2026-06-27.)**
- **Goal** = learn the policy that maximizes expected return — earlier-pitch decisions matter because they shape the rest of the at-bat
- **Evaluation** = **history-aware, return-based off-policy estimator** (see *Evaluation Methodology*). The earlier memoryless `(count, handedness, action)` reward-lookup table is **deprecated** as the primary metric — Phase 3 showed it structurally cannot credit sequence memory (any policy is capped by a memoryless reward-max oracle, so LSTM/Transformer "wins" only reflected reward-greedy selection, not memory). It is retained only as a sanity-check reference ceiling.

> **Why Option A:** an immediate per-pitch reward scored by a memoryless state-action table makes history invisible to both the reward and the evaluator, so the core research question (does sequence memory add value?) is untestable. A terminal/delayed reward credited back through the sequence makes the problem a genuine MDP where memory can earn its keep.

---

## Experiment Matrix

Full 3×3 design crossing three reward functions against three action spaces. Each of the 9 cells runs all 6 models (3 baselines + Bayesian Markov + LSTM + Transformer).

|  | Reward A — Whiff | Reward B — Weak Contact | Reward C — Combined |
|---|---|---|---|
| **Action 1 — Pitch Type Only** | Exp 1A | Exp 1B | Exp 1C |
| **Action 2 — Pitch Location Only** | Exp 2A | Exp 2B | Exp 2C |
| **Action 3 — Type + Location** | Exp 3A | Exp 3B | Exp 3C |

**Build order:** Complete Row 1 (type only) across all three reward functions first. Validate reward signal quality and model ranking before extending to Rows 2 and 3.

---

## Model Stack

Six models run in every experiment cell. Three establish the floor and ceiling; three are the primary competitors.

### Baselines (run first — establish floor and ceiling)

**Random Policy**
- Selects next pitch uniformly at random from the action space
- Absolute floor — anything that doesn't beat this is useless

**Empirical Frequency Baseline**
- Always selects the pitcher's most statistically common action in the current count + handedness state
- Surprisingly hard to beat — represents naive optimal without any sequence modeling
- If no model beats this consistently, sequencing history is not adding value

**Markov Chain**
- Transition matrix as a policy: P(next action | current state)
- Memoryless — ignores at-bat history beyond the current state
- Critical gating model: if LSTM and Transformer don't beat Markov, sequence memory is not justified

### Primary Models (the real competition)

**Bayesian Markov Chain** ← recommended co-primary
- Puts a Dirichlet prior on transition probabilities
- Handles data sparsity gracefully — critical for individual pitchers with limited at-bats in specific count states
- Natural uncertainty quantification on every recommendation — no additional work required
- If only one model beyond the baselines gets built first, this is it

**LSTM**
- Reads full at-bat sequence, outputs Q(s, a) per action
- Primary recurrent model — captures sequential dependencies across the at-bat
- Benchmark for whether sequence memory improves over memoryless Markov

**Transformer (self-attention)** ← recommended co-primary
- Attention mechanism over at-bat sequence rather than recurrence
- At-bats are short (3–7 pitches) — this favors Transformer over LSTM, which can struggle with vanishing gradients on short sequences
- No sequential bottleneck: attends to all prior pitches simultaneously
- Serious LSTM competitor; include from the start, not as an afterthought

### Secondary Models (informative, not primary)

These run after the primary stack is validated. Include in final paper but don't block on them.

| Model | Value |
|---|---|
| GRU | Lightweight LSTM alternative — if GRU matches LSTM, added LSTM complexity isn't justified |
| MLP (feedforward) | Flat concatenated input, no sequence modeling — tests whether sequence architecture matters at all |
| Bayesian Logistic Regression | Single-pitch interpretable baseline — if this matches sequence models, history isn't helping |
| HMM | Models at-bat as hidden intent states (attack / expand / waste) — maps onto how coaches think |

---

## Recommended Primary Comparison

**Bayesian Markov vs. LSTM vs. Transformer** — these three are the core research contribution.

- Bayesian Markov: handles sparsity + gives uncertainty for free
- LSTM: standard sequence memory benchmark
- Transformer: best-positioned NN for short sequence decision problems

If Bayesian Markov matches or beats both NNs, the finding is that probabilistic state modeling is sufficient and the teaching tool should be built on it — simpler, more interpretable, better uncertainty. If Transformer wins, that drives the platform architecture forward.

---

## Action Spaces

### Row 1 — Pitch Type Only
Action = discrete pitch type from pitcher's arsenal (FB, SI, CT, SL, CB, CH)
- Smallest action space — fastest to train and validate
- Answers: does pitch type sequencing order matter for outcomes?

### Row 2 — Pitch Location Only
Action = zone from 9-zone strike zone grid (plate_x / plate_z → zone mapping)
- Answers: does location sequencing matter independent of pitch type?
- Key insight test: if Row 2 outperforms Row 1, location > type for that reward function

### Row 3 — Pitch Type + Location
Action = combined (pitch_type, zone) pair
- Largest action space; requires most data
- Only attempted after Rows 1 and 2 validated
- Most realistic — mirrors actual pitcher decision-making

---

## Reward Functions

### Reward Structure — Terminal MDP (Option A)

Each at-bat is an **episode**. The terminal reward is credited at the **final pitch** of the at-bat (from `events`); the return for the decision at pitch `t` is `G_t = Σ_{k≥t} γ^(k−t) · r_k`. Discount `γ` = **0.9** (hyperparameter — **log it in every result output**). Models are trained to predict `Q(history-state, action) = E[G_t]`; policies recommend `argmax_a Q`.

**Design decision — terminal reward only in Phase 3 primary runs (2026-06-27):** the per-pitch shaping rewards (A/B/C tables below) are **deferred to a separate sensitivity run** done *after* the primary terminal-only results are confirmed. Rationale: running the **terminal signal alone** keeps the Phase-3 comparison clean and interpretable — any model separation is attributable purely to the terminal-outcome objective, not to a blend of shaping + terminal. So in Phase 3, `r_k = 0` for every non-terminal pitch and `r_T = terminal reward`, giving `G_t = γ^(T−t) · R_terminal`.

**Terminal PA-outcome reward** (CONFIRMED values, from `events`; shared across Rewards A/B/C — log them):

| Terminal PA outcome (`events`) | Reward |
|---|---|
| Strikeout | +1.5 |
| Out on contact (field_out, force_out, GIDP, double_play, sac_fly, sac_bunt, fielders_choice_out) | +0.8 |
| Single | -0.9 |
| Walk / HBP | -1.0 |
| XBH — Double / Triple | -1.3 |
| Home run | -1.6 |

Secondary mapping (not in the confirmed primary list): reached on error / fielders_choice → -0.5; `catcher_interf`, `truncated_pa` → excluded (incomplete/charged PA). Reward B's sensitivity run additionally grades terminal contact by exit velocity (weak / hard / barrel).

### Reward A — Whiff Rate (per-pitch shaping)
| Outcome | Reward |
|---|---|
| Swinging strike | +1.0 |
| Foul ball | +0.3 |
| Called strike | +0.2 |
| Ball | -0.3 |
| Contact (any) | -0.5 |
| Walk | -0.8 |

### Reward B — Weak Contact Rate
Requires `launch_speed` and `launch_angle` from Statcast.

| Outcome | Reward |
|---|---|
| Weak contact (exit velo < 85mph) | +1.0 |
| Called strike | +0.2 |
| Swinging strike | +0.2 |
| Ball / Walk | -0.5 |
| Hard contact (exit velo ≥ 95mph) | -1.0 |
| Barrel | -1.5 |

### Reward C — Combined
Weighted sum: `reward = (whiff_weight × whiff_signal) + (weak_contact_weight × contact_signal)`

Default weights: whiff = 1.0, weak contact = 0.7. Weights are hyperparameters — run sensitivity analysis. Always log weight config alongside results. Under Option A these weights scale the per-pitch shaping terms; the terminal PA-outcome reward is shared.

---

## Evaluation Methodology (Option A — history-aware, return-based)

Train on 2021–2022, evaluate on the 2023 holdout, no leakage. **Single source of truth:** one fixed direct-method estimator `Q*(s, a) = mean realized G_t` over training, with `s = (prev pitch, count, handedness)` — i.e. **one-step history-aware** (Markov-state granularity), a deliberate improvement over the deprecated memoryless `(count, hand)` table. Every model (baseline or primary) recommends an action per holdout decision; all are scored identically by `mean_holdout Q*(s_i, π(s_i))`. Sparse `(s,a)` cells back off to `(count, hand, a)` → `(a)` → global mean (track fallbacks).

- **NN models (LSTM / Transformer / GRU / MLP):** predict `Q(history-state, a) = E[G_t]` from the full pitch sequence via **Monte-Carlo return targets** (`G_t`, not TD); recommended action = `argmax_a Q`; uncertainty via MC-Dropout (30 passes). The network may use full history to *choose*; the score it earns is `Q*` at the recommended action.
- **Tabular baselines (Markov / Bayesian Markov):** condition on `(prev pitch, count, handedness)`, earning one-step history credit.
- **Empirical Frequency / Random:** unchanged as policies; scored on the same `Q*`.
- **Honest cap:** because `Q*` is one-step-history granularity, it credits memory up to one prior pitch. Beating the Markov family on `Q*` means *return-greedy action selection beats frequency-matching at the Markov-state level*; isolating value from **deeper** memory requires a separate analysis (e.g. whether NN recommendations diverge from the one-step return-max oracle in ways that track realized returns). Report this limitation, do not overclaim.
- **Reference ceiling (deprecated):** the memoryless `(count, hand, action)` table is computed only as a sanity check.

The headline comparison is **Transformer / LSTM vs. Bayesian Markov vs. Markov on expected `G_t`**: clear, honest separation (>~1 SE) is the bar; otherwise state-conditioned probabilistic modeling (Bayesian Markov) is sufficient and the teaching tool is built on it.

---

## Data

**Source:** MLB Statcast via `pybaseball`

```python
from pybaseball import statcast
df = statcast('2023-04-01', '2023-10-01')
```

| Column | Purpose |
|---|---|
| `pitch_type` | Action (Row 1 + 3), state |
| `plate_x`, `plate_z` | Action (Row 2 + 3), state, model input |
| `balls`, `strikes` | Count state |
| `stand` | Batter handedness |
| `description` | Reward A label (whiff / contact) |
| `events` | Terminal at-bat outcome |
| `launch_speed` | Reward B — exit velocity |
| `launch_angle` | Reward B — contact quality |
| `release_speed` | Model input feature |
| `pfx_x`, `pfx_z` | Movement — model input feature |
| `at_bat_number` | Sequence reconstruction |
| `pitch_number` | Position in sequence |

**Volume thresholds:**
- Minimum: 800 at-bats per pitcher
- Preferred: 1,500+ at-bats (2–3 seasons of starter data)
- Starters preferred over relievers
- Cache all Statcast pulls as local parquet files — never re-pull data already on disk

---

## Teaching Output

Each experiment cell produces:
- **Next pitch recommendation** — highest-value action given current state
- **Action ranking** — full ranked list with expected reward per option
- **Confidence score** — uncertainty estimate; flag low-confidence recommendations explicitly
- **Outcome probability curve** — how expected reward shifts as at-bat develops

**Cross-model agreement analysis:**
- Where all primary models agree → high-confidence teaching signal
- Where models disagree → surfaces strategic tradeoff between reward objectives
- Disagreement is not a failure — it is the most teachable output the system produces

---

## Tech Stack

- **Python 3.10+** — primary research language
- **pybaseball** — Statcast data retrieval
- **pandas / numpy** — data manipulation, sequence construction
- **scikit-learn** — preprocessing, evaluation metrics, logistic regression baseline
- **PyMC or Stan** — Bayesian Markov and Bayesian logistic regression
- **PyTorch** — LSTM, GRU, Transformer, MLP implementations
- **matplotlib / seaborn** — heatmaps, reward curves, action rankings
- **Jupyter Notebooks** — per-experiment documentation and reproducibility

---

## Documentation

- `README.md` — public-facing project overview, research design, findings, repo structure
- `docs/diagrams/bayesian_markov_architecture.png` — Bayesian Markov architecture and flow
- `docs/diagrams/lstm_architecture.png` — LSTM architecture and flow
- `docs/diagrams/transformer_architecture.png` — Transformer architecture and flow

- memory/ — point-in-time research state snapshots; 
- re-synced from ~/.claude/projects/... at each phase commit

---

## Key Constraints

- Individual pitcher models only — no cross-pitcher inference in early experiments
- All primary model outputs must include uncertainty estimates — point estimates alone not acceptable
- Reward weights in Experiment C are hyperparameters — always log config alongside results
- Do not assume pitch type dominates location — Row 2 results may challenge this
- Cache all Statcast data locally — never re-pull what is already on disk
- Secondary models (GRU, MLP, BLR, HMM) do not block primary model development

---

## Open Questions

- Unified vs. separate models for LHH and RHH? Start unified, split if performance diverges.
- Fine-tune from a league-wide base model or train per pitcher from scratch?
- What weight combination in Reward C best mirrors real at-bat winning conditions?
- Minimum at-bat threshold before a model is reliable enough for coaching use?
- How to surface Bayesian uncertainty in a way coaches can interpret without statistical background?

---

## Connection to Pathway Performance

- Next-pitch recommendations → coach dashboard sequencing tool (Phase 2)
- Agreement maps → program builder teaching content
- Reward curve visualizations → player-facing explainability screens
- Eventually: real-time sequencing suggestions during bullpen sessions (Phase 3)

---

## Status

| Item | Status |
|---|---|
| Architecture design | ✅ Complete |
| README.md | ✅ Complete |
| Architecture diagrams | ✅ Complete |
| Experiment matrix defined (3×3) | ✅ Complete |
| Reward functions defined | ✅ Complete — revised to terminal MDP (Option A) 2026-06-27 |
| Evaluation methodology (history-aware, return-based) | ✅ Defined (Option A) |
| Action spaces defined | ✅ Complete |
| Model stack defined | ✅ Complete |
| Data source confirmed (Statcast) | ✅ Complete (Gerrit Cole 2021–2023, cached) |
| Exp 1A — Type / Whiff (memoryless eval, Phase 2–3) | ⚠️ Built & run, but metric deprecated (memoryless) |
| Exp 1A — Type / Whiff (terminal reward, Option A) | ✅ Complete |
| Exp 1B — Type / Weak Contact | ✅ Complete |
| Exp 1C — Type / Combined | ✅ Complete |
| Row 1 memory files | ✅ Complete |
| Exp 2A — Location / Whiff | 🔲 Not started |
| Exp 2B — Location / Weak Contact | 🔲 Not started |
| Exp 2C — Location / Combined | 🔲 Not started |
| Exp 3A — Type+Location / Whiff | 🔲 Not started |
| Exp 3B — Type+Location / Weak Contact | 🔲 Not started |
| Exp 3C — Type+Location / Combined | 🔲 Not started |
| Row 1 cross-experiment agreement analysis (Phase 5) | ✅ Complete |
| Secondary models (GRU, MLP, BLR, HMM) | 🔲 Not started |
| Cross-model agreement analysis (Rows 2–3) | 🔲 Not started |
| Platform integration design | 🔲 Not started |

---

## Key Findings

**Row 1 (pitch type only — Exp 1A/1B/1C).** Bayesian Markov is the stable platform model; the LSTM/Transformer are unstable, degenerate, off-distribution, and fallback-propped on the type-only action space at ~1,600 at-bats (seed std 3–4× the eval SE; dominant pitch flips across seeds). NN architectural fixes (richer action space, per-pitch shaping, conservative-Q, league base model) are queued for Rows 2–3.

**Row 1 Phase 5 (agreement analysis).** Bayesian Markov is **reward-agnostic** on pitch-type-only — it recommends by transition frequency, so its pick is **identical across 1A/1B/1C for all 107 holdout states (100% agreement)**. The trustworthy teaching output is therefore a single reward-independent reference: 29 high-confidence states (CI width < 0.2, ≥20 holdout pitches), predominantly "throw the 4-seam," with a few slider/changeup states. **Revision to the earlier 18.6% disagreement finding:** that figure comes only from the reward-aware return-max *oracle*, and it is **sparse-cell noise, not a coachable signal** — its dominant case (0-0 vs LHH, 63% of all disagreement pitches) recommends a sinker for whiff based on just **2 training pitches**, and after requiring each pick to rest on ≥10 training pitches, **zero** trustworthy reward-differentiated states remain. **Conclusion:** the pitch-type-only action space cannot resolve the whiff-vs-weak-contact strategic question — the strongest motivation yet for Row 2 (location).
