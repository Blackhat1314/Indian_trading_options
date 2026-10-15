"""REST API of the research desks (AI analyst, Screener, Monte Carlo, Quant): /api/research/...

Placeholder while the research desks are built.
"""
from __future__ import annotations

from typing import Callable

from fastapi import APIRouter


def make_router(current: Callable, admin: Callable, csrf: Callable) -> APIRouter:
    return APIRouter(prefix="/api/research")
