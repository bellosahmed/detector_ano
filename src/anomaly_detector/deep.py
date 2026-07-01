"""Optional deep comparator — an LSTM autoencoder anomaly scorer.

This is a *comparison baseline*, not the default detector. The core system needs none of
this (no torch). It exists so we can honestly benchmark "does deep beat the classical
ensemble here?" on the same datasets and metrics. It only produces an anomaly *score*;
exact figures and explanations still come from code.

The autoencoder learns to reconstruct normal windows; windows it reconstructs poorly
(high error) are anomalous. Device-aware (CUDA if available) and batched so long series
never exhaust GPU memory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn


def get_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _window(values: np.ndarray, window: int) -> np.ndarray:
    """Sliding windows of length `window`, stride 1 → shape (n-window+1, window)."""
    n = len(values)
    if n < window:
        return values.reshape(1, -1)
    return np.lib.stride_tricks.sliding_window_view(values, window)


class LSTMAutoencoder(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.encoder = nn.LSTM(1, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: (B, W, 1)
        _, (h, _) = self.encoder(x)             # h: (1, B, hidden)
        z = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)  # (B, W, hidden)
        out, _ = self.decoder(z)
        return self.head(out)                   # (B, W, 1)


def lstm_autoencoder_scores(series: pd.Series, window: int = 32, hidden: int = 16,
                            epochs: int = 30, lr: float = 0.01, batch: int = 4096,
                            device: str | None = None, seed: int = 0) -> pd.Series:
    """Per-point anomaly score (reconstruction error) aligned to `series`.

    Higher = more anomalous. Missing values are interpolated; the series is standardised
    before windowing so the scale doesn't dominate training.
    """
    torch.manual_seed(seed)
    dev = get_device(device)

    values = pd.to_numeric(series, errors="coerce").interpolate(limit_direction="both").to_numpy(float)
    mu, sd = np.nanmean(values), np.nanstd(values) or 1.0
    norm = (values - mu) / sd

    wins = _window(norm, window)                      # (num_windows, window)
    x = torch.tensor(wins, dtype=torch.float32, device=dev).unsqueeze(-1)  # (Nw, W, 1)

    model = LSTMAutoencoder(hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(x.size(0), device=dev)
        for i in range(0, x.size(0), batch):
            xb = x[perm[i:i + batch]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()

    # batched inference → per-window mean squared reconstruction error
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            xb = x[i:i + batch]
            errs.append(((model(xb) - xb) ** 2).mean(dim=(1, 2)).cpu().numpy())
    win_err = np.concatenate(errs) if errs else np.zeros(1)

    # assign each window's error to its centre, then fill the edges
    scores = np.full(len(values), np.nan)
    centre = window // 2
    for j, e in enumerate(win_err):
        scores[min(j + centre, len(values) - 1)] = e
    out = pd.Series(scores, index=series.index).interpolate(limit_direction="both")
    return out.fillna(0.0)
