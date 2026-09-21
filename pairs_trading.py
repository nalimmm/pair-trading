"""
Stat arb / pairs trading - full pipeline.

Methodology:
  1. Download daily prices (8 years) for 4 US equity pairs in the same sector.
  2. On the TRAIN window (70%): test for cointegration (Engle-Granger) and
     estimate the hedge ratio beta via OLS(A ~ B).
  3. On the TEST window (30%, out-of-sample): build the spread with the
     frozen beta, compute its z-score (rolling 60d, estimated only on TEST,
     no leakage from TRAIN), and trade a mean-reversion signal:
       z > +2    -> short spread (short A, long beta*B)
       z < -2    -> long spread  (long A, short beta*B)
       |z| < 0.5 -> flat
  4. Apply transaction costs (5 bps per leg, on every position change).
  5. Compute annualized Sharpe, max drawdown, turnover.

Universe (same sector, close economic exposure):
  - XOM / CVX   (integrated energy)
  - KO  / PEP   (defensive consumer / beverages)
  - JPM / BAC   (US too-big-to-fail banks)
  - HD  / LOW   (home improvement retail)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

PAIRS = [("XOM", "CVX"), ("KO", "PEP"), ("JPM", "BAC"), ("HD", "LOW")]
PERIOD = "8y"
TRAIN_FRAC = 0.70
Z_ENTRY = 2.0
Z_EXIT = 0.5
ROLL_WINDOW = 60
COST_BPS = 5 / 10000  # per leg, per position change
ANNUALIZATION = 252


def download_prices(tickers):
    data = yf.download(list(set(tickers)), period=PERIOD, progress=False)["Close"]
    return data.dropna()


def backtest_pair(prices, a, b):
    px = prices[[a, b]].dropna()
    n = len(px)
    split = int(n * TRAIN_FRAC)
    train, test = px.iloc[:split], px.iloc[split:]

    # Cointegration test on TRAIN
    score, pvalue, _ = coint(train[a], train[b])

    # Hedge ratio beta estimated on TRAIN only
    X = sm.add_constant(train[b])
    model = sm.OLS(train[a], X).fit()
    beta = model.params[b]

    # Spread + z-score on TEST, rolling, without using TRAIN
    spread_test = test[a] - beta * test[b]
    roll_mean = spread_test.rolling(ROLL_WINDOW).mean()
    roll_std = spread_test.rolling(ROLL_WINDOW).std()
    zscore = (spread_test - roll_mean) / roll_std

    # Position: -1 short spread, +1 long spread, 0 flat
    position = pd.Series(0.0, index=test.index)
    pos = 0
    for i, z in enumerate(zscore):
        if np.isnan(z):
            position.iloc[i] = 0
            continue
        if pos == 0:
            if z > Z_ENTRY:
                pos = -1
            elif z < -Z_ENTRY:
                pos = 1
        else:
            if abs(z) < Z_EXIT:
                pos = 0
        position.iloc[i] = pos

    # PnL: spread return = return(A) - beta * return(B)
    ret_a = test[a].pct_change()
    ret_b = test[b].pct_change()
    spread_ret = ret_a - beta * ret_b

    # position applied with a 1-day lag (no look-ahead)
    strat_ret = position.shift(1) * spread_ret

    # Transaction costs: on every position change, across both legs
    turnover_series = position.diff().abs().fillna(0)
    costs = turnover_series * COST_BPS * (1 + abs(beta))
    strat_ret_net = strat_ret - costs

    strat_ret_net = strat_ret_net.dropna()
    cum = (1 + strat_ret_net).cumprod()

    sharpe = (strat_ret_net.mean() / strat_ret_net.std()) * np.sqrt(ANNUALIZATION) if strat_ret_net.std() > 0 else np.nan
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = drawdown.min()
    turnover = turnover_series.sum() / len(turnover_series) * ANNUALIZATION  # trades per year (approx)

    return {
        "pair": f"{a}/{b}",
        "coint_pvalue": pvalue,
        "beta": beta,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "annual_turnover": turnover,
        "total_return": cum.iloc[-1] - 1,
        "n_days_test": len(test),
        "cum_series": cum,
    }


def main():
    all_tickers = [t for pair in PAIRS for t in pair]
    prices = download_prices(all_tickers)

    results = []
    for a, b in PAIRS:
        res = backtest_pair(prices, a, b)
        results.append(res)

    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "cum_series"} for r in results])
    summary = summary.set_index("pair")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    summary.to_csv("results_summary.csv")

    # save cumulative curves for the report
    curves = pd.DataFrame({r["pair"]: r["cum_series"] for r in results})
    curves.to_csv("results_curves.csv")

    return summary, curves


if __name__ == "__main__":
    main()
