"""Black-Scholes-Merton pricing, Greeks and implied volatility."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def _check(kind):
    if kind not in ("call", "put"):
        raise ValueError("option type must be 'call' or 'put'")


def bs_price(S, K, T, r, sigma, kind="call", q=0.0) -> float:
    _check(kind)
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    d1, d2 = _d(S, K, T, r, sigma, q)
    if kind == "call":
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float   # per 1 vol point (1%)
    theta: float  # per calendar day
    rho: float    # per 1% rate move


def greeks(S, K, T, r, sigma, kind="call", q=0.0) -> Greeks:
    _check(kind)
    d1, d2 = _d(S, K, T, r, sigma, q)
    pdf, eq, er = norm.pdf(d1), np.exp(-q * T), np.exp(-r * T)
    common = -S * eq * pdf * sigma / (2 * np.sqrt(T))
    if kind == "call":
        delta, theta = eq * norm.cdf(d1), common - r * K * er * norm.cdf(d2) + q * S * eq * norm.cdf(d1)
        rho = K * T * er * norm.cdf(d2)
    else:
        delta, theta = eq * (norm.cdf(d1) - 1), common + r * K * er * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1)
        rho = -K * T * er * norm.cdf(-d2)
    return Greeks(float(delta), float(eq * pdf / (S * sigma * np.sqrt(T))), float(S * eq * pdf * np.sqrt(T) / 100),
                  float(theta / 365), float(rho / 100))


def implied_vol(price, S, K, T, r, kind="call", q=0.0) -> float:
    _check(kind)
    lo_bound = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if kind == "call" else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    if price <= lo_bound + 1e-12:
        raise ValueError("price is at/below intrinsic value: no implied volatility exists")
    try:
        return float(brentq(lambda s: bs_price(S, K, T, r, s, kind, q) - price, 1e-4, 5.0, xtol=1e-10))
    except ValueError as exc:
        raise ValueError("could not solve for implied volatility (price outside the 0.01%-500% vol range)") from exc
