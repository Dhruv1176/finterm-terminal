"""The interactive shell."""
from __future__ import annotations

import cmd
import shlex

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__, analytics as an, charts, data, pricing, state as st


class UsageError(ValueError):
    pass


BANNER = r"""[bold cyan]
  _____ _       _____                   
 |  ___(_)_ __ |_   _|__ _ __ _ __ ___  
 | |_  | | '_ \  | |/ _ \ '__| '_ ` _ \ 
 |  _| | | | | | | |  __/ |  | | | | | |
 |_|   |_|_| |_| |_|\___|_|  |_| |_| |_|[/]"""

COMMANDS = [
    ("quote T [T..]", "Price, 1D/5D/1M change, 52-week position, trend sparkline"),
    ("chart T [days]", "Price chart in the terminal (default 120 days)"),
    ("watch [add|rm T..]", "Show or edit your watchlist"),
    ("buy T qty [price]", "Record a purchase (price defaults to last)"),
    ("sell T qty [price]", "Record a sale; realised P&L is tracked"),
    ("portfolio", "Positions, P&L, weights"),
    ("risk", "Portfolio volatility, VaR/CVaR, drawdown, beta"),
    ("optimize T T [T..]", "Min-volatility and max-Sharpe weights"),
    ("corr T T [T..]", "Correlation heat-map"),
    ("mc T [days] [paths]", "Monte Carlo price simulation (GBM)"),
    ("bs call|put S K T r% vol% [q%]", "Black-Scholes price + Greeks (T in years)"),
    ("iv call|put S K T r% price", "Implied volatility from a market price"),
    ("loan amount rate% years", "EMI and amortisation summary"),
    ("sip monthly rate% years", "Systematic investment plan projection"),
    ("benchmark T", "Set the beta benchmark (default SPY)"),
    ("demo on|off", "Toggle synthetic demo data"),
    ("exit", "Quit"),
]


def money(x: float) -> str:
    return f"{x:,.2f}"


def pct(x: float, plus: bool = True) -> Text:
    return Text(f"{x:+.2%}" if plus else f"{x:.2%}", style="green" if x >= 0 else "red")


def _num(s: str, name: str) -> float:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        raise UsageError(f"{name} must be a number, got {s!r}")


class FinTerm(cmd.Cmd):
    prompt = "finterm> "

    def __init__(self, console: Console | None = None):
        super().__init__()
        self.console = console or Console()
        self.state = st.load()

    # ---------- plumbing ----------
    def onecmd(self, line):
        try:
            return super().onecmd(line)
        except UsageError as e:
            self.console.print(f"[yellow]Usage:[/] {escape(str(e))}")
        except ValueError as e:
            self.console.print(f"[red]Error:[/] {escape(str(e))}")
        except Exception as e:  # keep the shell alive no matter what
            self.console.print(f"[red]Unexpected {type(e).__name__}:[/] {escape(str(e))}")

    def emptyline(self):
        pass

    def default(self, line):
        self.console.print(f"[yellow]Unknown command:[/] {line.split()[0]}  (type [bold]help[/])")

    def _args(self, arg, minimum=0, usage=""):
        a = shlex.split(arg)
        if len(a) < minimum:
            raise UsageError(usage)
        return a

    def _px(self, t, days=756):
        h = data.get_history(t, days)
        if data.STATE["reason"]:
            self.console.print(f"[yellow]Note: {data.STATE['reason']}[/]")
            data.STATE["reason"] = ""
        return h

    def _save(self):
        st.save(self.state)

    def preloop(self):
        self.console.print(BANNER)
        self.console.print(f"  v{__version__} · data: [bold]{data.source()}[/] · type [bold]help[/] for commands\n")

    # ---------- market ----------
    def _quote_table(self, tickers):
        wide = self.console.width >= 100
        t = Table(box=box.SIMPLE_HEAVY, header_style="bold")
        cols = [("Ticker", "left"), ("Last", "right"), ("1D", "right"), ("5D", "right"), ("1M", "right")]
        cols += [("52W range", "left")] if wide else []
        cols += [("Trend (60d)", "left")]
        for c, j in cols:
            t.add_column(c, justify=j, no_wrap=True)
        for tk in tickers:
            s = self._px(tk, 260)
            last, ch = s.iloc[-1], lambda k: s.iloc[-1] / s.iloc[-1 - k] - 1
            lo, hi = s.min(), s.max()
            pos = int(round((last - lo) / ((hi - lo) or 1) * 10))
            bar = "[dim]" + "─" * pos + "[/][bold]●[/][dim]" + "─" * (10 - pos) + "[/]"
            tr = Text(charts.sparkline(s.iloc[-60:].values, 20), style="green" if s.iloc[-1] >= s.iloc[-60] else "red")
            cells = [f"[bold]{tk}[/]", money(last), pct(ch(1)), pct(ch(5)), pct(ch(21))]
            cells += [f"{lo:,.0f} {bar} {hi:,.0f}"] if wide else []
            t.add_row(*cells, tr)
        self.console.print(t)

    def do_quote(self, arg):
        tk = [x.upper() for x in self._args(arg, 1, "quote TICKER [TICKER ...]")]
        self._quote_table(tk)

    def do_chart(self, arg):
        a = self._args(arg, 1, "chart TICKER [days]")
        days = int(_num(a[1], "days")) if len(a) > 1 else 120
        s = self._px(a[0].upper(), max(days, 10))
        up = s.iloc[-1] >= s.iloc[0]
        color = "green" if up else "red"
        body = "\n".join(charts.area_chart(s.values, width=max(20, min(70, len(s), self.console.width - 18)), height=12))
        self.console.print(Panel(Text(body, style=color), title=f"{a[0].upper()} · last {len(s)} sessions · {s.iloc[-1] / s.iloc[0] - 1:+.2%}",
                                 subtitle=f"{s.index[0].date()} → {s.index[-1].date()}", border_style=color))

    def do_watch(self, arg):
        a = self._args(arg)
        if a and a[0] in ("add", "rm"):
            if len(a) < 2:
                raise UsageError("watch add|rm TICKER [TICKER ...]")
            for tk in (x.upper() for x in a[1:]):
                if a[0] == "add":
                    self._px(tk, 30)  # validates the ticker
                    if tk not in self.state["watchlist"]:
                        self.state["watchlist"].append(tk)
                elif tk in self.state["watchlist"]:
                    self.state["watchlist"].remove(tk)
            self._save()
        if not self.state["watchlist"]:
            self.console.print("[dim]Watchlist is empty. Try: watch add AAPL[/]")
        else:
            self._quote_table(self.state["watchlist"])

    # ---------- portfolio ----------
    def do_buy(self, arg):
        a = self._args(arg, 2, "buy TICKER QTY [PRICE]")
        tk, q = a[0].upper(), _num(a[1], "qty")
        p = _num(a[2], "price") if len(a) > 2 else float(self._px(tk, 5).iloc[-1])
        st.buy(self.state, tk, q, p)
        self._save()
        self.console.print(f"[green]Bought[/] {q:g} {tk} @ {money(p)}")

    def do_sell(self, arg):
        a = self._args(arg, 2, "sell TICKER QTY [PRICE]")
        tk, q = a[0].upper(), _num(a[1], "qty")
        p = _num(a[2], "price") if len(a) > 2 else float(self._px(tk, 5).iloc[-1])
        r = st.sell(self.state, tk, q, p)
        self._save()
        self.console.print(f"[green]Sold[/] {q:g} {tk} @ {money(p)} · realised P&L ", pct(r / max(q * p, 1e-9)), f"({money(r)})")

    def do_portfolio(self, arg):
        pos = self.state["positions"]
        if not pos:
            self.console.print("[dim]No positions yet. Try: buy AAPL 10[/]")
            return
        rows = {tk: (p["qty"], p["cost"], float(self._px(tk, 5).iloc[-1])) for tk, p in pos.items()}
        total = sum(q * l for q, _, l in rows.values())
        t = Table(box=box.SIMPLE_HEAVY, header_style="bold")
        for c in ("Ticker", "Qty", "Avg cost", "Last", "Value", "P&L", "P&L %", "Weight"):
            t.add_column(c, justify="left" if c == "Ticker" else "right")
        cost_total = 0.0
        for tk, (q, c, l) in rows.items():
            cost_total += q * c
            t.add_row(f"[bold]{tk}[/]", f"{q:g}", money(c), money(l), money(q * l),
                      Text(money((l - c) * q), style="green" if l >= c else "red"), pct(l / c - 1), f"{q * l / total:.1%}")
        t.add_section()
        t.add_row("[bold]TOTAL[/]", "", "", "", f"[bold]{money(total)}[/]", Text(money(total - cost_total), style="green" if total >= cost_total else "red"),
                  pct(total / cost_total - 1), "100%")
        self.console.print(t)
        self.console.print(f"Realised P&L to date: [bold]{money(self.state['realized'])}[/]")

    def _portfolio_returns(self):
        pos = self.state["positions"]
        if not pos:
            raise ValueError("no positions yet (use buy)")
        px = pd.concat({tk: self._px(tk, 504) for tk in pos}, axis=1).dropna()
        val = pd.Series({tk: pos[tk]["qty"] * px[tk].iloc[-1] for tk in pos})
        w = val / val.sum()
        return (px.pct_change().dropna() * w).sum(axis=1), val.sum()

    def do_risk(self, arg):
        r, total = self._portfolio_returns()
        v95, c95 = an.var_cvar(r, 0.95)
        v99, _ = an.var_cvar(r, 0.99)
        eq = (1 + r).cumprod()
        bench = self.state["benchmark"]
        b = an.beta(r, an.returns(self._px(bench, 504)))
        t = Table(title=f"Portfolio risk · value {money(total)} · {len(r)} days", box=box.SIMPLE_HEAVY, header_style="bold")
        t.add_column("Metric")
        t.add_column("Value", justify="right")
        t.add_column("In money", justify="right")
        t.add_row("Annualised volatility", f"{r.std() * np.sqrt(an.TD):.2%}", "")
        t.add_row("VaR 95% (1 day)", f"{v95:.2%}", money(v95 * total))
        t.add_row("VaR 99% (1 day)", f"{v99:.2%}", money(v99 * total))
        t.add_row("CVaR 95% (1 day)", f"{c95:.2%}", money(c95 * total))
        t.add_row("VaR 95% (10 day, √t)", f"{v95 * np.sqrt(10):.2%}", money(v95 * np.sqrt(10) * total))
        t.add_row("Max drawdown", f"{an.max_drawdown(eq):.2%}", "")
        t.add_row(f"Beta vs {bench}", f"{b:.2f}", "")
        self.console.print(t)

    def do_benchmark(self, arg):
        a = self._args(arg, 1, "benchmark TICKER")
        self._px(a[0].upper(), 30)
        self.state["benchmark"] = a[0].upper()
        self._save()
        self.console.print(f"Benchmark set to [bold]{a[0].upper()}[/]")

    def do_optimize(self, arg):
        tk = [x.upper() for x in self._args(arg, 2, "optimize T1 T2 [T3 ...]")]
        rets = pd.concat({t: an.returns(self._px(t, 756)) for t in tk}, axis=1).dropna()
        res = an.optimal_weights(rets)
        t = Table(title="Long-only optimal weights (3y daily data, rf 4%)", box=box.SIMPLE_HEAVY, header_style="bold")
        t.add_column("Portfolio")
        for x in tk:
            t.add_column(x, justify="right")
        t.add_column("Return", justify="right")
        t.add_column("Vol", justify="right")
        for name, (w, mu, vol) in res.items():
            t.add_row(name, *[f"{x:.0%} " + "█" * int(x * 10) for x in w], f"{mu:.1%}", f"{vol:.1%}")
        self.console.print(t)
        self.console.print("[dim]In-sample estimates: expect worse out-of-sample. See the portfolio-risk-optimizer project for walk-forward tests.[/]")

    def do_corr(self, arg):
        tk = [x.upper() for x in self._args(arg, 2, "corr T1 T2 [T3 ...]")]
        c = pd.concat({t: an.returns(self._px(t, 504)) for t in tk}, axis=1).dropna().corr()
        t = Table(title="Correlation of daily returns (2y)", box=box.SIMPLE_HEAVY)
        t.add_column("")
        for x in tk:
            t.add_column(x, justify="right")
        for a in tk:
            cells = []
            for b in tk:
                v = c.loc[a, b]
                g = int(abs(v) * 140)
                cells.append(Text(f"{v:+.2f}", style=f"bold white on rgb({0 if v >= 0 else g + 40},{g + 40 if v >= 0 else 0},0)"))
            t.add_row(f"[bold]{a}[/]", *cells)
        self.console.print(t)

    def do_mc(self, arg):
        a = self._args(arg, 1, "mc TICKER [days=252] [paths=5000]")
        tk = a[0].upper()
        days = int(_num(a[1], "days")) if len(a) > 1 else 252
        n = int(_num(a[2], "paths")) if len(a) > 2 else 5000
        s = self._px(tk, 756)
        lr = np.log(s).diff().dropna()
        mu, sig = lr.mean() * an.TD + 0.5 * lr.var() * an.TD, lr.std() * np.sqrt(an.TD)
        paths = an.gbm_paths(float(s.iloc[-1]), mu, sig, days, n)
        end = paths[:, -1]
        q = np.percentile(end, [5, 25, 50, 75, 95])
        t = Table(title=f"{tk} Monte Carlo · {n:,} paths · {days} days · σ={sig:.1%} μ={mu:.1%}", box=box.SIMPLE_HEAVY, header_style="bold")
        for c in ("5th", "25th", "Median", "75th", "95th", "P(loss)", "Expected"):
            t.add_column(c, justify="right")
        t.add_row(*[money(x) for x in q], f"{(end < s.iloc[-1]).mean():.1%}", money(end.mean()))
        self.console.print(t)
        self.console.print(Panel("\n".join(charts.histogram(end, width=max(10, min(34, self.console.width - 50)))), title="Distribution of terminal price", border_style="cyan"))

    # ---------- derivatives & personal finance ----------
    def do_bs(self, arg):
        a = self._args(arg, 6, "bs call|put SPOT STRIKE YEARS RATE% VOL% [DIV%]")
        kind, (S, K, T, r, v) = a[0].lower(), [_num(x, n) for x, n in zip(a[1:6], ("spot", "strike", "years", "rate", "vol"))]
        q = _num(a[6], "dividend") / 100 if len(a) > 6 else 0.0
        r, v = r / 100, v / 100
        px, g = pricing.bs_price(S, K, T, r, v, kind, q), pricing.greeks(S, K, T, r, v, kind, q)
        t = Table(title=f"Black-Scholes {kind.upper()} · S={S:g} K={K:g} T={T:g}y r={r:.2%} σ={v:.2%}", box=box.SIMPLE_HEAVY)
        t.add_column("Price", justify="right", style="bold")
        for c in ("Delta", "Gamma", "Vega /1%", "Theta /day", "Rho /1%", "Moneyness"):
            t.add_column(c, justify="right")
        t.add_row(f"{px:,.4f}", f"{g.delta:.4f}", f"{g.gamma:.5f}", f"{g.vega:.4f}", f"{g.theta:.4f}", f"{g.rho:.4f}", f"{S / K:.1%}")
        self.console.print(t)
        spots = np.linspace(0.6 * S, 1.4 * S, 60)
        vals = [pricing.bs_price(x, K, T, r, v, kind, q) for x in spots]
        self.console.print(Panel(Text("\n".join(charts.area_chart(vals, max(20, min(60, self.console.width - 18)), 8)), style="cyan"),
                                 title=f"Option value vs spot ({spots[0]:,.0f} → {spots[-1]:,.0f})", border_style="cyan"))

    def do_iv(self, arg):
        a = self._args(arg, 6, "iv call|put SPOT STRIKE YEARS RATE% PRICE")
        kind, (S, K, T, r, p) = a[0].lower(), [_num(x, n) for x, n in zip(a[1:6], ("spot", "strike", "years", "rate", "price"))]
        iv = pricing.implied_vol(p, S, K, T, r / 100, kind)
        self.console.print(f"Implied volatility: [bold cyan]{iv:.2%}[/]  (reprice check: {pricing.bs_price(S, K, T, r / 100, iv, kind):.4f})")

    def do_loan(self, arg):
        a = self._args(arg, 3, "loan AMOUNT RATE% YEARS")
        P, rate, yrs = _num(a[0], "amount"), _num(a[1], "rate"), _num(a[2], "years")
        pay = an.emi(P, rate, yrs)
        sched = an.amortization(P, rate, yrs)
        total_int = sched["interest_paid"].sum()
        self.console.print(Panel(f"EMI [bold cyan]{money(pay)}[/] / month\nTotal interest [bold]{money(total_int)}[/]  ·  total paid [bold]{money(P + total_int)}[/]  ·  interest is [bold]{total_int / P:.0%}[/] of principal",
                                 title="Loan summary", border_style="cyan"))
        t = Table(box=box.SIMPLE_HEAVY, header_style="bold")
        for c in ("Year", "Principal paid", "Interest paid", "Balance", "Interest share"):
            t.add_column(c, justify="right")
        for _, r in sched.iterrows():
            tot = r.principal_paid + r.interest_paid
            t.add_row(str(int(r.year)), money(r.principal_paid), money(r.interest_paid), money(r.balance), f"{r.interest_paid / tot:.0%} " + "▇" * int(r.interest_paid / tot * 10))
        self.console.print(t)

    def do_sip(self, arg):
        a = self._args(arg, 3, "sip MONTHLY RATE% YEARS")
        m, rate, yrs = _num(a[0], "monthly"), _num(a[1], "rate"), _num(a[2], "years")
        fv, inv = an.sip_future_value(m, rate, yrs), m * round(yrs * 12)
        curve = [an.sip_future_value(m, rate, y / 4) for y in range(1, int(yrs * 4) + 1)]
        self.console.print(Panel(f"Invested [bold]{money(inv)}[/] → worth [bold cyan]{money(fv)}[/]\nGain [bold green]{money(fv - inv)}[/] ({fv / inv - 1:.0%}) · wealth multiple [bold]{fv / inv:.2f}x[/]",
                                 title=f"SIP · {money(m)}/month · {rate:g}% · {yrs:g} years", border_style="cyan"))
        self.console.print(Text("\n".join(charts.area_chart(curve, max(20, min(64, len(curve) * 2, self.console.width - 16)), 10, "{:,.0f}")), style="green"))

    # ---------- system ----------
    def do_demo(self, arg):
        a = self._args(arg, 1, "demo on|off")
        data.set_demo(a[0].lower() == "on")
        self.console.print(f"Data source: [bold]{data.source()}[/]")

    def do_help(self, arg):
        t = Table(title="FinTerm commands", box=box.SIMPLE_HEAVY, header_style="bold")
        t.add_column("Command", style="cyan")
        t.add_column("What it does")
        for c, d in COMMANDS:
            t.add_row(escape(c), d)
        self.console.print(t)

    def do_exit(self, arg):
        self.console.print("[dim]Goodbye.[/]")
        return True

    do_quit = do_EOF = do_exit
