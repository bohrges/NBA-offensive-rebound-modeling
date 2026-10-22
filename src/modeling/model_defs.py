### model_defs.py
### Hybrid LSTM + MLP architecture, shared by modeling_lstm.ipynb and evaluate_calibration.py.
### FRAME_SIZE = ball(x,y,z) + 5 OFF(x,y,vx,vy) + 5 DEF(x,y,vx,vy).

import torch
import torch.nn as nn

FRAME_SIZE = 43


# Additive (Bahdanau-style) attention over LSTM time steps.
# Returns a weighted context vector and the attention weights for interpretability.
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out, pad_mask=None):
        # lstm_out : (B, T, H)
        # pad_mask : (B, T) bool - True = padded (to be ignored)
        scores = self.attn(lstm_out).squeeze(-1)              # (B, T)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, float('-inf'))
        weights = torch.softmax(scores, dim=1)                # (B, T)
        context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # (B, H)
        return context, weights


# Hybrid LSTM + MLP model for rebound type prediction.
#
# Parameters:
# static_input  : int   - number of static tabular features (STATIC_DIM)
# lstm_hidden   : int   - LSTM hidden size
# lstm_layers   : int   - number of stacked LSTM layers
# mlp_hidden    : int   - static branch hidden size
# fusion_hidden : int   - fusion head hidden size
# dropout       : float - dropout rate (applied in all sub-networks)
class HybridModel(nn.Module):
    def __init__(self, static_input, lstm_hidden=64, lstm_layers=2,
                 mlp_hidden=64, fusion_hidden=64, dropout=0.3):
        super().__init__()

        # - Sequence branch -
        self.lstm = nn.LSTM(
            input_size  = FRAME_SIZE,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0
        )
        self.attention = TemporalAttention(lstm_hidden)

        # - Static branch -
        self.static_branch = nn.Sequential(
            nn.Linear(static_input, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # - Fusion head -
        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden + mlp_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1)   # raw logit (BCEWithLogitsLoss)
        )

    # x_seq      : (B, MAX_FRAMES, FRAME_SIZE) - left-padded
    # seq_lens   : (B,)                        - true lengths from meta
    # x_static   : (B, STATIC_DIM)
    # return_attn: bool - if True, also return attention weights (B, MAX_FRAMES)
    def forward(self, x_seq, seq_lens, x_static, return_attn=False):
        B, T, _ = x_seq.shape

        # Left-padding mask: True = padded frame (first T - seq_len positions)
        idx      = torch.arange(T, device=x_seq.device).unsqueeze(0)   # (1, T)
        pad_mask = idx < (T - seq_lens.unsqueeze(1))                    # (B, T)

        lstm_out, _        = self.lstm(x_seq)                           # (B, T, H)
        context, attn_w    = self.attention(lstm_out, pad_mask)         # (B, H), (B, T)

        static_out = self.static_branch(x_static)                       # (B, mlp_hidden)
        fused      = torch.cat([context, static_out], dim=1)            # (B, H + mlp_hidden)
        logits     = self.fusion(fused).squeeze(1)                      # (B,)

        return (logits, attn_w) if return_attn else logits


# Construct a HybridModel from a params dict (Optuna/baseline compatible).
def build_model(params: dict, static_input: int, device="cpu"):
    return HybridModel(
        static_input  = static_input,
        lstm_hidden   = params['lstm_hidden'],
        lstm_layers   = params['lstm_layers'],
        mlp_hidden    = params['mlp_hidden'],
        fusion_hidden = params['fusion_hidden'],
        dropout       = params['dropout']
    ).to(device)