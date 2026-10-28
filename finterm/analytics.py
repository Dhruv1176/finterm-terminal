"""Pure analytics functions (no I/O), all unit-tested."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TD = 252


def returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def var_cvar(r, alpha: float = 0.95):
    r = np.asarray(r)
    q = np.quantile(r, 1 - alpha)
    return float(-q), float(-r[r <= q].mean())


def max_drawdown(prices) -> float:
    p = pd.Series(prices)
    return float((p / p.cummax() - 1).min())


def beta(asset: pd.Series, bench: pd.Series) -> float:
    d = pd.concat([asset, bench], axis=1).dropna()
    return float(d.cov().iloc[0, 1] / d.iloc[:, 1].var())


def gbm_paths(s0: float, mu: float, sigma: float, days: int, paths: int, seed: int | None = 0) -> np.ndarray:
    """Geometric Brownian motion; mu/sigma are annualised arithmetic drift and vol. Shape (paths, days+1)."""
    rng = np.random.default_rng(seed)
    dt = 1 / TD
    z = rng.standard_normal((paths, days))
    steps = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    return s0 * np.exp(np.concatenate([np.zeros((paths, 1)), np.cumsum(steps, axis=1)], axis=1))


def emi(principal: float, annual_rate_pct: float, years: float) -> float:
    n, i = int(round(years * 12)), annual_rate_pct / 1200
    return principal / n if i == 0 else principal * i * (1 + i) ** n / ((1 + i) ** n - 1)


def amortization(principal: float, annual_rate_pct: float, years: float) -> pd.DataFrame:
    """Yearly roll-up of a fixed-rate amortising loan."""
    n, i, pay = int(round(years * 12)), annual_rate_pct / 1200, emi(principal, annual_rate_pct, years)
    bal, rows = principal, []
    ip = pp = 0.0
    for m in range(1, n + 1):
        interest = bal * i
        princ = pay - interest
        bal -= princ
        ip, pp = ip + interest, pp + princ
        if m % 12 == 0 or m == n:
            rows.append({"year": int(np.ceil(m / 12)), "principal_paid": pp, "interest_paid": ip, "balance": max(bal, 0.0)})
            ip = pp = 0.0
    return pd.DataFrame(rows)


def sip_future_value(monthly: float, annual_rate_pct: float, years: float) -> float:
    """Future value of a monthly investment made at the start of each month."""
    n, i = int(round(years * 12)), annual_rate_pct / 1200
    return monthly * n if i == 0 else monthly * (((1 + i) ** n - 1) / i) * (1 + i)


def optimal_weights(rets: pd.DataFrame, rf: float = 0.04):
    """Long-only min-variance and max-Sharpe weights from daily returns."""
    mu, cov, n = rets.mean().values * TD, rets.cov().values * TD, rets.shape[1]
    cons, bnds, x0 = [{"type": "eq", "fun": lambda w: w.sum() - 1}], [(0, 1)] * n, np.full(n, 1 / n)

    def solve(f):
        w = np.clip(minimize(f, x0, method="SLSQP", bounds=bnds, constraints=cons).x, 0, None)
        return w / w.sum()

    wmv = solve(lambda w: w @ cov @ w)
    wms = solve(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w))
    stats = lambda w: (float(w @ mu), float(np.sqrt(w @ cov @ w)))
    return {"Min volatility": (wmv, *stats(wmv)), "Max Sharpe": (wms, *stats(wms))}
