"""The networks (PyTorch — needed only on the training PC; the server runs the exported ONNX files).

Daily (NeuralEdgeNet), 3 heads: up after 1 day, 1 week, 1 month
  seq (B, 40, 26)  ── linear → GRU(hidden) ── last hidden state ─┐
  seq[:, -1]       ── today's 26 features (skip path) ───────────┤
  static (B, S)    ── linear(static_hidden) → GELU → dropout ────┼─ concat → linear(64) → GELU → dropout → heads

09:30 → close (IntradayNet), 2 heads: up by the close, beat the day's median stock
  seq, static      the daily branch above, as of the previous close
  bars (B, 15, 7)  ── linear → GRU(bar_hidden) ── last hidden state ┐  the first 15 one-minute bars
  bars[:, -1]      ── the 09:29 bar (skip path) ─────────────────────┤
  intra (B, I)     ── linear(static_hidden) → GELU → dropout ────────┴─ concat with the daily branch → heads

Small on purpose (under ~100k weights): a stock panel holds a few hundred thousand rows but only a few
dozen independent market swings (or, intraday, a thousand-odd trading days), and a bigger network memorises them.
Several seeds are trained and averaged; the sizes can be searched (td.neural.train, --search).
"""
from __future__ import annotations

import torch
from torch import nn


class NeuralEdgeNet(nn.Module):
    def __init__(self, n_tech: int, n_static: int, n_heads: int = 3, hidden: int = 48, static_hidden: int = 64,
                 dropout: float = 0.25, input_dropout: float = 0.1, **_):
        super().__init__()
        self.in_drop = nn.Dropout(input_dropout)
        self.proj = nn.Linear(n_tech, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.static = nn.Sequential(nn.Linear(n_static, static_hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden + static_hidden + n_tech, 64), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(64, n_heads))

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        s = self.in_drop(seq)
        _, h = self.gru(torch.tanh(self.proj(s)))
        z = torch.cat([h[-1], self.static(static), s[:, -1, :]], dim=1)
        return self.head(z)                                   # logits, one per horizon


class IntradayNet(nn.Module):
    def __init__(self, n_tech: int, n_static: int, n_bar: int, n_intra: int, n_heads: int = 2, hidden: int = 48,
                 static_hidden: int = 64, bar_hidden: int = 32, dropout: float = 0.25, input_dropout: float = 0.1,
                 **_):
        super().__init__()
        self.in_drop = nn.Dropout(input_dropout)
        self.proj = nn.Linear(n_tech, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.static = nn.Sequential(nn.Linear(n_static, static_hidden), nn.GELU(), nn.Dropout(dropout))
        self.bproj = nn.Linear(n_bar, bar_hidden)
        self.bgru = nn.GRU(bar_hidden, bar_hidden, batch_first=True)
        self.intra = nn.Sequential(nn.Linear(n_intra, static_hidden), nn.GELU(), nn.Dropout(dropout))
        width = hidden + static_hidden + n_tech + bar_hidden + n_bar + static_hidden
        self.head = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, n_heads))

    def forward(self, seq: torch.Tensor, static: torch.Tensor, bars: torch.Tensor,
                intra: torch.Tensor) -> torch.Tensor:
        s = self.in_drop(seq)
        _, h = self.gru(torch.tanh(self.proj(s)))
        b = self.in_drop(bars)
        _, hb = self.bgru(torch.tanh(self.bproj(b)))
        z = torch.cat([h[-1], self.static(static), s[:, -1, :], hb[-1], b[:, -1, :], self.intra(intra)], dim=1)
        return self.head(z)


class Ensemble(nn.Module):
    """Mean probability of several trained seeds — what is exported to ONNX."""

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.sigmoid(m(*inputs)) for m in self.members]).mean(dim=0)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
