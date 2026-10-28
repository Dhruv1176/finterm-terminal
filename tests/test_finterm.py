import io

import numpy as np
import pytest
from rich.console import Console

from finterm import analytics as an, charts, data, pricing, state as st
from finterm.shell import FinTerm


def test_put_call_parity():
    S, K, T, r, v, q = 100, 95, 0.75, 0.04, 0.25, 0.01
    c, p = pricing.bs_price(S, K, T, r, v, "call", q), pricing.bs_price(S, K, T, r, v, "put", q)
    assert np.isclose(c - p, S * np.exp(-q * T) - K * np.exp(-r * T))


def test_known_bs_value():  # Hull textbook: S=42 K=40 r=10% sigma=20% T=0.5 -> call 4.76
    assert round(pricing.bs_price(42, 40, 0.5, 0.10, 0.20, "call"), 2) == 4.76


def test_greeks_match_finite_differences():
    S, K, T, r, v = 100, 100, 1.0, 0.05, 0.2
    g, h = pricing.greeks(S, K, T, r, v, "call"), 1e-3
    f = lambda s=S, vol=v, t=T: pricing.bs_price(s, K, t, r, vol, "call")
    assert np.isclose(g.delta, (f(S + h) - f(S - h)) / (2 * h), atol=1e-5)
    assert np.isclose(g.gamma, (f(S + h) - 2 * f() + f(S - h)) / h**2, atol=1e-4)
    assert np.isclose(g.vega, (f(vol=v + 1e-4) - f(vol=v - 1e-4)) / 2e-4 / 100, atol=1e-4)
    assert np.isclose(g.theta, (f(t=T - 1 / 365) - f()), atol=2e-3)


def test_implied_vol_roundtrip_and_errors():
    p = pricing.bs_price(100, 110, 0.5, 0.03, 0.31, "put")
    assert np.isclose(pricing.implied_vol(p, 100, 110, 0.5, 0.03, "put"), 0.31, atol=1e-6)
    with pytest.raises(ValueError):
        pricing.implied_vol(0.0, 100, 110, 0.5, 0.03, "call")


def test_emi_and_amortisation():
    assert round(an.emi(100000, 12, 1), 2) == 8884.88
    s = an.amortization(500000, 8.5, 20)
    assert np.isclose(s["principal_paid"].sum(), 500000) and s["balance"].iloc[-1] < 1e-6


def test_sip():
    assert np.isclose(an.sip_future_value(1000, 0, 2), 24000)
    assert an.sip_future_value(1000, 12, 10) > 1000 * 120


def test_var_cvar_and_gbm():
    r = np.random.default_rng(0).normal(0, 0.01, 100000)
    v, c = an.var_cvar(r, 0.95)
    assert c > v and abs(v - 0.01645) < 5e-4
    paths = an.gbm_paths(100, 0.08, 0.2, 252, 20000)
    assert paths.shape == (20000, 253) and abs(paths[:, -1].mean() / 100 - np.exp(0.08)) < 0.02


def test_optimal_weights_sum_to_one():
    import pandas as pd
    rets = pd.concat({t: an.returns(data.synthetic_history(t)) for t in ("AAA", "BBB", "CCC")}, axis=1).dropna()
    for w, _, _ in an.optimal_weights(rets).values():
        assert np.isclose(w.sum(), 1) and (w >= 0).all()


def test_charts():
    lines = charts.area_chart(np.linspace(1, 10, 50), width=40, height=8)
    assert len(lines) == 9 and all("│" in l or "└" in l for l in lines)
    assert len(charts.sparkline(np.arange(100), 20)) == 20


def test_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("FINTERM_HOME", str(tmp_path))
    s = st.load()
    st.buy(s, "X", 10, 100)
    st.buy(s, "X", 10, 120)
    assert s["positions"]["X"] == {"qty": 20, "cost": 110}
    assert st.sell(s, "X", 5, 130) == 100
    with pytest.raises(ValueError):
        st.sell(s, "X", 99, 1)
    st.save(s)
    assert st.load()["positions"]["X"]["qty"] == 15


def test_shell_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("FINTERM_HOME", str(tmp_path))
    data.set_demo(True)
    buf = io.StringIO()
    sh = FinTerm(Console(file=buf, width=140, force_terminal=False))
    for c in ["quote AAPL MSFT", "chart AAPL 90", "buy AAPL 10", "buy MSFT 5 400", "portfolio", "risk", "optimize AAPL MSFT SPY",
              "corr AAPL MSFT SPY", "mc AAPL 126 2000", "bs call 100 105 0.5 5 20", "iv put 100 110 0.5 3 12",
              "loan 5000000 8.5 20", "sip 10000 12 15", "watch add TSLA", "sell AAPL 99", "bogus", "bs call 1"]:
        sh.onecmd(c)
    out = buf.getvalue()
    for needle in ["AAPL", "TOTAL", "VaR 95%", "Max Sharpe", "Correlation", "Monte Carlo", "Black-Scholes", "Implied volatility",
                   "EMI", "SIP", "Error:", "Unknown command", "Usage:"]:
        assert needle in out, needle
    assert "Traceback" not in out and "Unexpected" not in out
