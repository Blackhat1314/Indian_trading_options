"""The network (PyTorch — needed only on the training PC; the server runs the exported ONNX file).

  seq (B, 40, 26)  ── linear → GRU(48) ── last hidden state ─┐
  seq[:, -1]       ── today's 26 features (skip path) ───────┤
  static (B, S)    ── linear(64) → GELU → dropout ───────────┼─ concat → linear(64) → GELU → dropout → 3 logits
                                                             ┘                              (1 day, 1 week, 1 month)

Small on purpose (~25k weights): a daily stock panel holds a few hundred thousand rows but only a few dozen
independent market swings, and a bigger network memorises them. Several seeds are trained and averaged.
"""
from __future__ import annotations

import torch
from torch import nn


class NeuralEdgeNet(nn.Module):
    def __init__(self, n_tech: int, n_static: int, n_heads: int = 3, hidden: int = 48, static_hidden: int = 64,
                 dropout: float = 0.25, input_dropout: float = 0.1):
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


class Ensemble(nn.Module):
    """Mean P(up) of several trained seeds — what is exported to ONNX."""

    def __init__(self, members: list[NeuralEdgeNet]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.sigmoid(m(seq, static)) for m in self.members]).mean(dim=0)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
