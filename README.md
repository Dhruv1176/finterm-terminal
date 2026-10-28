# FinTerm: a finance workstation for your terminal

![CI](https://github.com/YOUR_USERNAME/finterm/actions/workflows/ci.yml/badge.svg)

One command, a whole toolkit: live quotes, terminal charts, a persistent portfolio with P&L and risk, options pricing
with Greeks, Monte Carlo simulation, portfolio optimisation, and loan/SIP calculators. Built with Python and
[Rich](https://github.com/Textualize/rich). Works offline in demo mode.

<!-- Record a 20-second demo with `vhs` or `asciinema`, save as docs/demo.gif, and uncomment: -->
<!-- ![demo](docs/demo.gif) -->

## Install
```bash
pip install -e ".[live]"        # live market data via yfinance
finterm                         # interactive shell
finterm --demo quote AAPL MSFT  # one-shot command, offline synthetic data
```

## Commands
| Command | What it does |
|---|---|
| `quote AAPL MSFT` | Last, 1D/5D/1M change, 52-week position, 60-day sparkline |
| `chart AAPL 180` | Filled price chart drawn with Unicode blocks |
| `watch add TSLA` / `watch rm TSLA` | Persistent watchlist |
| `buy AAPL 10` / `sell AAPL 5 190` | Positions with weighted-average cost and realised P&L |
| `portfolio` | Holdings, unrealised P&L, weights |
| `risk` | Volatility, 1-day and 10-day VaR/CVaR in money terms, max drawdown, beta vs benchmark |
| `optimize AAPL MSFT JPM GLD` | Min-volatility and max-Sharpe weights |
| `corr AAPL MSFT GLD` | Colour-coded correlation matrix |
| `mc AAPL 252 5000` | GBM Monte Carlo: percentile table, P(loss), terminal-price histogram |
| `bs call 100 105 0.5 5 20` | Black-Scholes price + Delta/Gamma/Vega/Theta/Rho and payoff-vs-spot chart |
| `iv call 100 105 0.5 5 6.2` | Implied volatility (Brent root-finding) |
| `loan 5000000 8.5 20` | EMI, total interest, yearly amortisation |
| `sip 10000 12 15` | Monthly-investment projection |

Rates and volatilities are entered in percent (`5` means 5%); time to expiry in years.
State lives in `~/.finterm/state.json` (override with `FINTERM_HOME`).
Indian markets work with Yahoo tickers: `finterm quote RELIANCE.NS TCS.NS` and `benchmark ^NSEI`.

## Under the hood
- `pricing.py`: Black-Scholes-Merton with continuous dividend yield; Greeks verified against finite differences; implied vol via `brentq`; put-call parity and a textbook value tested.
- `analytics.py`: pure, unit-tested functions: historical VaR/CVaR, beta, GBM simulation, amortisation, SIP future value, SLSQP optimiser.
- `charts.py`: area charts at 1/8-block resolution, sparklines, histograms.
- `data.py`: yfinance with automatic, clearly-flagged fallback to deterministic synthetic prices that share a common market factor (so correlation, beta and diversification behave sensibly).
- Robust shell: friendly usage errors, never crashes on bad input; smoke-tested end to end.

```bash
python -m pytest -q    # 11 tests
```

## Roadmap
Options chain viewer · binomial American options · alerts · CSV import of trades · FX / crypto · Textual full-screen TUI

MIT License. Educational use only; not investment advice. Synthetic demo prices are not real market data.

## Author

**Dhruv Singh**

GitHub: [@Dhruv1176](https://github.com/Dhruv1176)
