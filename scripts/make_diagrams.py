"""Generate architecture flowchart PNGs for the three primary models.

Renders programmatically with matplotlib (Agg backend) to docs/diagrams/.
Each image: white background, ~1200px wide, title + caption + side annotation.
"""
from __future__ import annotations

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "diagrams")
os.makedirs(OUT, exist_ok=True)

# palette
C_STATE = "#dbeafe"   # light blue
C_PROC = "#e0e7ff"    # indigo
C_MODEL = "#fef3c7"   # amber
C_OUT = "#dcfce7"     # green
C_NOTE = "#fde2e4"    # rose (annotation / failure)
C_NOTE_OK = "#e7f5ec" # green note
EDGE = "#334155"


def _box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.6, edgecolor=EDGE, facecolor=color, mutation_aspect=1))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color="#0f172a")
    if sub:
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=9.2, color="#1e293b")


def _arrow(ax, x, y0, y1, color=EDGE, style="-|>"):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle=style, mutation_scale=18,
                                 linewidth=1.8, color=color))


def _side_note(ax, x, y, w, h, heading, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.4, edgecolor="#9f1239" if color == C_NOTE else "#15803d",
                                facecolor=color))
    ax.text(x + w / 2, y + h - 0.035, heading, ha="center", va="top",
            fontsize=11, fontweight="bold", color="#0f172a")
    wrapped = "\n".join(textwrap.fill(line, 34) for line in body.split("\n"))
    ax.text(x + 0.018, y + h - 0.10, wrapped, ha="left", va="top", fontsize=9.0, color="#1f2937")


def base_fig(height_in, title, caption):
    fig = plt.figure(figsize=(12, height_in), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="white", zorder=-10))
    ax.text(0.5, 0.975, title, ha="center", va="top", fontsize=18, fontweight="bold", color="#0f172a")
    ax.text(0.5, 0.018, caption, ha="center", va="bottom", fontsize=10, style="italic", color="#475569")
    return fig, ax


def stack(ax, boxes, x=0.07, w=0.52, top=0.90, bottom=0.085):
    n = len(boxes); h = 0.072
    gap = (top - bottom - n * h) / (n - 1)
    ys = []
    y = top - h
    for i, (t, s, c) in enumerate(boxes):
        _box(ax, x, y, w, h, t, s, c); ys.append((y, h))
        if i < n - 1:
            _arrow(ax, x + w / 2, y - 0.004, y - gap + 0.004)
        y -= (h + gap)
    return ys, x, w


# --------------------------------------------------------------------------
def diagram_bayes():
    fig, ax = base_fig(13, "Bayesian Markov Chain — Architecture & Flow",
                       "Row-1 platform model: a Dirichlet-smoothed transition policy that returns a "
                       "pitch recommendation with a calibrated 90% credible interval.")
    boxes = [
        ("At-Bat State", "last pitch type · count · batter handedness", C_STATE),
        ("Transition Matrix", "P(next pitch | state)  ·  from 2021-2022 training", C_PROC),
        ("Dirichlet Prior  (alpha = 1, symmetric)", "smooths sparse count-state cells", C_PROC),
        ("Posterior Distribution", "a full distribution per action — not a point estimate", C_MODEL),
        ("Policy:  argmax posterior mean", "pick the highest-probability next pitch", C_MODEL),
        ("Recommendation", "pitch type + 90% credible interval  (flag if CI width > 0.3)", C_OUT),
    ]
    stack(ax, boxes)
    _side_note(ax, 0.64, 0.40, 0.32, 0.34, "Why Bayesian Markov for Row 1",
               "• Stable & reproducible — no random-seed lottery.\n"
               "• Interpretable — a transition table a coach can read.\n"
               "• Handles sparsity — the Dirichlet prior fills thin "
               "count-state cells gracefully.\n"
               "• Uncertainty for free — every pick ships with a "
               "credible interval; low-confidence states are flagged, "
               "not hidden.\n\n"
               "Result: ties the empirical optimum on Exp 1A and is the "
               "chosen Row-1 teaching model.", C_NOTE_OK)
    fig.savefig(os.path.join(OUT, "bayesian_markov_architecture.png"), facecolor="white")
    plt.close(fig)


def diagram_lstm():
    fig, ax = base_fig(15, "LSTM — Architecture & Flow",
                       "A recurrent value model: reads the at-bat pitch-by-pitch, carrying sequence "
                       "memory forward, and outputs an expected-return Q-value per pitch type.")
    boxes = [
        ("At-Bat Sequence", "pitch 1 -> pitch 2 -> pitch 3 -> ... -> pitch N", C_STATE),
        ("Input Vector per Pitch", "type one-hot · plate_x · plate_z · balls · strikes · velo · pfx_x · pfx_z", C_STATE),
        ("LSTM Layer 1   (hidden 64, dropout 0.2)", "cell state carries sequence memory forward", C_PROC),
        ("LSTM Layer 2   (hidden 64, dropout 0.2)", "deeper temporal features", C_PROC),
        ("Final Hidden State", "encodes the full at-bat history up to pitch N", C_MODEL),
        ("Output Head", "Q(s, a) — expected return for each pitch type", C_MODEL),
        ("MC-Dropout Inference", "30 stochastic forward passes -> mean +/- std", C_PROC),
        ("Policy:  argmax mean Q  + uncertainty", "next-pitch recommendation with confidence", C_OUT),
    ]
    ys, x, w = stack(ax, boxes)
    # recurrent self-loop on LSTM Layer 1 (3rd box, index 2)
    yb, hb = ys[2]
    ax.add_patch(FancyArrowPatch((x + w, yb + hb * 0.5), (x + w + 0.05, yb + hb * 0.5),
                                 connectionstyle="arc3,rad=-1.4", arrowstyle="-|>",
                                 mutation_scale=15, linewidth=1.8, color="#7c3aed"))
    ax.text(x + w + 0.085, yb + hb * 0.5, "recurrent\nh(t-1) -> h(t)", ha="left", va="center",
            fontsize=8.6, color="#7c3aed", fontweight="bold")
    _side_note(ax, 0.64, 0.32, 0.32, 0.30, "Known failure mode (Row 1)",
               "On pitch-type-only at ~1,600 at-bats the LSTM is "
               "UNSTABLE: across random seeds E[G_t] swings widely and "
               "the dominant recommended pitch flips (FC/FF/SL/KC).\n\n"
               "It collapses to degenerate, off-distribution policies "
               "(~42% cutters vs Cole's ~4%).\n\n"
               "Fix queued for Rows 2-3: richer action space + "
               "behavior-regularized (conservative-Q) training.", C_NOTE)
    fig.savefig(os.path.join(OUT, "lstm_architecture.png"), facecolor="white")
    plt.close(fig)


def diagram_transformer():
    fig, ax = base_fig(15.5, "Transformer (Self-Attention) — Architecture & Flow",
                       "An attention value model: all prior pitches are read simultaneously; attention "
                       "weights reveal which earlier pitches drove the next-pitch recommendation.")
    # top: parallel sequence boxes feeding attention simultaneously
    px, pw, py, ph = 0.07, 0.52, 0.855, 0.06
    n = 4
    cellw = (pw - 0.03 * (n - 1)) / n
    labels = ["pitch 1", "pitch 2", "pitch 3", "pitch N"]
    centers = []
    for i, lab in enumerate(labels):
        cx = px + i * (cellw + 0.03)
        ax.add_patch(FancyBboxPatch((cx, py), cellw, ph, boxstyle="round,pad=0.008,rounding_size=0.015",
                                    linewidth=1.4, edgecolor=EDGE, facecolor=C_STATE))
        ax.text(cx + cellw / 2, py + ph / 2, lab, ha="center", va="center", fontsize=10, fontweight="bold")
        centers.append(cx + cellw / 2)
    ax.text(px + pw / 2, py + ph + 0.022, "At-Bat Sequence — all pitches available at once (no recurrence)",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#0f172a")

    boxes = [
        ("Input Embedding per Pitch", "same feature vector as LSTM  +  positional encoding (pitch position)", C_STATE),
        ("Multi-Head Self-Attention  (4 heads)", "each head learns which prior pitches to weight", C_PROC),
        ("Feedforward Layer  (dim 128, dropout 0.1)", "per-position non-linear transform", C_PROC),
        ("Encoder Block 2  (attention + feedforward)", "second stacked layer", C_PROC),
        ("Output Head", "Q(s, a) — expected return for each pitch type", C_MODEL),
        ("MC-Dropout Inference", "30 stochastic forward passes -> mean +/- std", C_PROC),
        ("Policy:  argmax mean Q  + attention map", "recommendation + which pitches mattered", C_OUT),
    ]
    ys, x, w = stack(ax, boxes, top=0.80, bottom=0.085)
    # arrows from each parallel pitch box into the embedding box (simultaneous)
    eb_y = ys[0][0] + ys[0][1]
    for cx in centers:
        ax.add_patch(FancyArrowPatch((cx, py - 0.004), (x + w / 2, eb_y + 0.004),
                                     arrowstyle="-|>", mutation_scale=12, linewidth=1.2,
                                     color="#64748b", connectionstyle="arc3,rad=0.0"))
    # attention -> explainability branch
    yb, hb = ys[1]
    ax.add_patch(FancyArrowPatch((x + w, yb + hb * 0.5), (0.64, yb + hb * 0.5),
                                 arrowstyle="-|>", mutation_scale=15, linewidth=1.8, color="#0e7490"))
    _side_note(ax, 0.64, yb - 0.10, 0.32, 0.20, "Attention = explainability layer",
               "The attention weights show which prior pitches in the "
               "at-bat the model weighted most for its recommendation — "
               "the coaching 'why'.", C_NOTE_OK)
    _side_note(ax, 0.64, 0.34, 0.32, 0.20, "Why attention for short at-bats",
               "All pitches feed in simultaneously, so there is no "
               "recurrent bottleneck — better suited than LSTM to short "
               "(3-7 pitch) sequences where vanishing gradients bite.", C_NOTE_OK)
    fig.savefig(os.path.join(OUT, "transformer_architecture.png"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    diagram_bayes()
    diagram_lstm()
    diagram_transformer()
    for f in ("bayesian_markov_architecture.png", "lstm_architecture.png", "transformer_architecture.png"):
        p = os.path.join(OUT, f)
        print(f"wrote {p}  ({os.path.getsize(p)} bytes)")
