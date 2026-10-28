"""Persistent watchlist / positions stored as JSON in ~/.finterm (override with FINTERM_HOME)."""
from __future__ import annotations

import json
import os
from pathlib import Path


def _path() -> Path:
    home = Path(os.environ.get("FINTERM_HOME", Path.home() / ".finterm"))
    home.mkdir(parents=True, exist_ok=True)
    return home / "state.json"


DEFAULT = {"watchlist": ["AAPL", "MSFT", "SPY"], "positions": {}, "realized": 0.0, "benchmark": "SPY"}


def load() -> dict:
    p = _path()
    if p.exists():
        try:
            return {**DEFAULT, **json.loads(p.read_text())}
        except json.JSONDecodeError:
            pass
    return json.loads(json.dumps(DEFAULT))


def save(state: dict) -> None:
    _path().write_text(json.dumps(state, indent=2))


def buy(state: dict, ticker: str, qty: float, price: float) -> None:
    if qty <= 0 or price <= 0:
        raise ValueError("quantity and price must be positive")
    pos = state["positions"].get(ticker, {"qty": 0.0, "cost": 0.0})
    total = pos["qty"] + qty
    pos["cost"] = (pos["qty"] * pos["cost"] + qty * price) / total   # weighted average cost
    pos["qty"] = total
    state["positions"][ticker] = pos


def sell(state: dict, ticker: str, qty: float, price: float) -> float:
    pos = state["positions"].get(ticker)
    if not pos or qty > pos["qty"] + 1e-9:
        raise ValueError(f"cannot sell {qty:g} {ticker}: you hold {pos['qty'] if pos else 0:g}")
    realized = (price - pos["cost"]) * qty
    state["realized"] += realized
    pos["qty"] -= qty
    if pos["qty"] <= 1e-9:
        del state["positions"][ticker]
    return realized
