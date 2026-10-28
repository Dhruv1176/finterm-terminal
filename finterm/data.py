"""Market data: live via yfinance when available, otherwise deterministic synthetic prices (demo mode)."""
from __future__ import annotations

import os
import zlib

import numpy as np
import pandas as pd

_N = 1500  # synthetic history length (business days)
_CACHE: dict[str, pd.Series] = {}
STATE = {"demo": os.environ.get("FINTERM_DEMO") == "1", "reason": ""}


def set_demo(flag: bool, reason: str = "") -> None:
    STATE["demo"], STATE["reason"] = flag, reason
    _CACHE.clear()


def source() -> str:
    return "DEMO (synthetic)" if STATE["demo"] else "LIVE (Yahoo Finance)"


def synthetic_history(ticker: str) -> pd.Series:
    """Every ticker shares one market factor (so correlations, beta and diversification behave realistically)."""
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=_N)
    market = np.random.default_rng(2024).normal(0.0004, 0.009, _N)
    r = np.random.default_rng(zlib.crc32(ticker.encode()))
    if ticker.upper() in ("SPY", "^GSPC", "^NSEI"):
        beta, vol, drift = 1.0, 0.0015, 0.0
    else:
        beta, vol, drift = r.uniform(0.5, 1.6), r.uniform(0.007, 0.02), r.uniform(-0.0001, 0.0006)
    ret = drift + beta * market + r.normal(0, vol, _N)
    start = 40 + (zlib.crc32(ticker.encode()) % 460)
    return pd.Series(start * np.exp(np.cumsum(ret)), index=idx, name=ticker.upper())


def get_history(ticker: str, days: int = 756) -> pd.Series:
    t = ticker.upper()
    if t not in _CACHE:
        if STATE["demo"]:
            _CACHE[t] = synthetic_history(t)
        else:
            try:
                import yfinance as yf

                s = yf.download(t, period="5y", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
                if len(s) < 30:
                    raise RuntimeError(f"no data for {t}")
                s.name = t
                _CACHE[t] = s
            except Exception as exc:
                if "no data" in str(exc):
                    raise ValueError(f"No data found for ticker {t!r}.") from exc
                set_demo(True, f"live data unavailable ({type(exc).__name__}); switched to demo data")
                return get_history(t, days)
    return _CACHE[t].iloc[-days:]
