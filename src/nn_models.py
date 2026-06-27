"""LSTM and Transformer Q(s,a) models for Experiment 1A (pitch type / Reward A).

Decision framing (causal, leakage-free)
---------------------------------------
One training example per pitch = one decision. For the decision at pitch t the
model sees the fully-realized history of pitches 1..t-1 plus a QUERY token that
carries only pre-pitch information (current count + handedness). It predicts
Q(s_t, a) for each action; only the taken action's head is supervised toward the
realized Reward A (masked MSE — direct/batch off-policy value regression). The
query token never sees pitch t's location/movement, so there is no leakage of the
action being chosen.

SI binning: for these neural models SI is binned with FF (5-action space), as
decided in Phase 2. The Bayesian Markov keeps SI separate.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Action space for the NN models: SI binned into FF -> 5 classes.
NN_ACTIONS = ["FF", "SL", "KC", "CH", "FC"]
NN_ACTION_IDX = {a: i for i, a in enumerate(NN_ACTIONS)}
_SI_BIN = {"SI": "FF"}  # documented SI->FF binning

NUMERIC_COLS = ["plate_x", "plate_z", "release_speed", "pfx_x", "pfx_z"]
MAX_LEN = 11  # max pitches per at-bat in the data
# token layout: 5 type onehot + [plate_x,plate_z,balls,strikes,release,pfx_x,pfx_z]
#               + 2 stand onehot + 1 is_query  = 15
TOKEN_DIM = 5 + 7 + 2 + 1


def bin_type(pt: str) -> str:
    return _SI_BIN.get(pt, pt)


def compute_feat_stats(train: pd.DataFrame) -> dict:
    return {c: (float(train[c].mean()), float(train[c].std() + 1e-8)) for c in NUMERIC_COLS}


def _pitch_token(row, stats, is_query, stand):
    """Build one token. If is_query, type/location/movement are zeroed and only
    the pre-pitch count + handedness are populated."""
    tok = np.zeros(TOKEN_DIM, dtype=np.float32)
    if not is_query:
        a = bin_type(row["pitch_type"])
        tok[NN_ACTION_IDX[a]] = 1.0
        tok[5] = (row["plate_x"] - stats["plate_x"][0]) / stats["plate_x"][1]
        tok[6] = (row["plate_z"] - stats["plate_z"][0]) / stats["plate_z"][1]
        tok[9] = (row["release_speed"] - stats["release_speed"][0]) / stats["release_speed"][1]
        tok[10] = (row["pfx_x"] - stats["pfx_x"][0]) / stats["pfx_x"][1]
        tok[11] = (row["pfx_z"] - stats["pfx_z"][0]) / stats["pfx_z"][1]
    # count (pre-pitch) is known at decision time for both history & query tokens
    tok[7] = row["balls"] / 3.0
    tok[8] = row["strikes"] / 2.0
    tok[12] = 1.0 if stand == "L" else 0.0
    tok[13] = 1.0 if stand == "R" else 0.0
    tok[14] = 1.0 if is_query else 0.0
    return tok


def build_examples(df: pd.DataFrame, stats: dict, target_col: str = "reward_a"):
    """Build per-decision padded sequences from at-bat sequences.

    Returns dict of tensors: X [N, MAX_LEN, TOKEN_DIM], lengths [N],
    action_idx [N], rewards [N] (= target_col, e.g. 'G_t'), plus:
      states      : (count_state, stand)            — memoryless eval key
      eval_states : (prev_pitch, count_state, stand) — history-aware G_t eval key
    """
    X, lengths, actions, rewards, states, eval_states = [], [], [], [], [], []
    for ab_id, ab in df.groupby("ab_id", sort=False):
        ab = ab.sort_values("pitch_number")
        rows = ab.to_dict("records")
        stand = rows[0]["stand"]
        for t in range(len(rows)):
            hist = [_pitch_token(rows[j], stats, False, stand) for j in range(t)]
            query = _pitch_token(rows[t], stats, True, stand)
            seq = hist + [query]
            L = len(seq)
            arr = np.zeros((MAX_LEN, TOKEN_DIM), dtype=np.float32)
            arr[:L] = np.array(seq, dtype=np.float32)
            X.append(arr)
            lengths.append(L)
            actions.append(NN_ACTION_IDX[bin_type(rows[t]["pitch_type"])])
            rewards.append(float(rows[t][target_col]))
            states.append((rows[t]["count_state"], stand))
            prev = rows[t - 1]["pitch_type"] if t > 0 else "NONE"
            eval_states.append((prev, rows[t]["count_state"], stand))
    return {
        "X": torch.tensor(np.array(X)),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "actions": torch.tensor(actions, dtype=torch.long),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "states": states,
        "eval_states": eval_states,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class LSTMQ(nn.Module):
    def __init__(self, n_actions=5, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(TOKEN_DIM, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, x, lengths):
        out, _ = self.lstm(x)                       # [B, L, H]
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
        last = out.gather(1, idx).squeeze(1)        # output at query token
        return self.head(self.drop(last))           # [B, n_actions]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class _EncoderLayer(nn.Module):
    """Single transformer encoder layer exposing attention weights."""
    def __init__(self, d_model, heads, ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(ff, d_model))
        self.drop = nn.Dropout(dropout)
        self.last_attn = None

    def forward(self, x, key_padding_mask=None):
        a, w = self.attn(x, x, x, key_padding_mask=key_padding_mask,
                         need_weights=True, average_attn_weights=True)
        self.last_attn = w  # [B, L, L] averaged over heads
        x = self.norm1(x + self.drop(a))
        x = self.norm2(x + self.ff(x))
        return x


class TransformerQ(nn.Module):
    def __init__(self, n_actions=5, d_model=64, heads=4, ff=128, layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(TOKEN_DIM, d_model)
        # learned positional encoding over pitch position within the at-bat
        self.pos_emb = nn.Embedding(MAX_LEN, d_model)
        self.layers = nn.ModuleList([_EncoderLayer(d_model, heads, ff, dropout)
                                     for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_actions)

    def forward(self, x, lengths, return_attn=False):
        B, L, _ = x.shape
        pad_mask = torch.arange(L).unsqueeze(0) >= lengths.unsqueeze(1)  # True = pad
        pos = torch.arange(L).unsqueeze(0).expand(B, L)
        h = self.proj(x) + self.pos_emb(pos)
        for layer in self.layers:
            h = layer(h, key_padding_mask=pad_mask)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(2))
        query = h.gather(1, idx).squeeze(1)  # query-token representation
        q = self.head(self.drop(query))
        if return_attn:
            # attention from the query token over the sequence, last layer
            last_w = self.layers[-1].last_attn  # [B, L, L]
            qi = (lengths - 1)
            attn = last_w[torch.arange(B), qi]  # [B, L] query attends to positions
            return q, attn
        return q


# ---------------------------------------------------------------------------
# Training / inference utilities
# ---------------------------------------------------------------------------
def masked_mse(pred, action_idx, reward):
    chosen = pred.gather(1, action_idx.view(-1, 1)).squeeze(1)
    return ((chosen - reward) ** 2).mean()


def train_model(model, train_ex, val_ex, max_epochs=150, patience=12, lr=1e-3,
                batch_size=64, seed=0, verbose=False):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(train_ex["actions"])
    best_val, best_state, wait = float("inf"), None, 0
    history = []
    g = torch.Generator().manual_seed(seed)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(train_ex["X"][b], train_ex["lengths"][b])
            loss = masked_mse(pred, train_ex["actions"][b], train_ex["rewards"][b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        train_loss = tot / n
        model.eval()
        with torch.no_grad():
            vp = model(val_ex["X"], val_ex["lengths"])
            val_loss = masked_mse(vp, val_ex["actions"], val_ex["rewards"]).item()
        history.append((epoch, train_loss, val_loss))
        if verbose:
            print(f"    epoch {epoch:3d} train={train_loss:.4f} val={val_loss:.4f}")
        if val_loss < best_val - 1e-5:
            best_val, best_state, wait = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val


def predict_actions(model, ex):
    """Deterministic (dropout off) argmax recommendation per decision."""
    model.eval()
    with torch.no_grad():
        q = model(ex["X"], ex["lengths"])
    idx = q.argmax(dim=1).numpy()
    return [NN_ACTIONS[i] for i in idx], q.numpy()


def _enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout(model, ex, passes=30, seed=0):
    """Keep dropout active; return per-decision mean & std of the chosen action's
    Q across `passes`, with the chosen action from the deterministic pass."""
    torch.manual_seed(seed)
    rec_actions, _ = predict_actions(model, ex)
    rec_idx = torch.tensor([NN_ACTION_IDX[a] for a in rec_actions])
    model.eval()
    _enable_dropout(model)
    samples = []
    with torch.no_grad():
        for _ in range(passes):
            q = model(ex["X"], ex["lengths"])
            samples.append(q.gather(1, rec_idx.view(-1, 1)).squeeze(1).numpy())
    samples = np.stack(samples)  # [passes, N]
    return rec_actions, samples.mean(0), samples.std(0)
