#!/usr/bin/env python3
"""Backtesting engine for BTC cycle strategies."""
import pandas as pd
import numpy as np
from typing import Callable

def load_data(path="backtest/btc_daily.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df

def backtest(df: pd.DataFrame, strategy_fn: Callable, initial_capital=10000.0, name="Strategy"):
    """
    Run backtest. strategy_fn(df) should return a Series of signals:
      1 = buy, -1 = sell, 0 = hold
    """
    signals = strategy_fn(df)
    capital = initial_capital
    position = 0.0  # BTC held
    entry_price = 0.0
    trades = []
    equity = []
    in_position = False

    for i in range(len(df)):
        price = df["close"].iloc[i]
        date = df.index[i]

        if signals.iloc[i] == 1 and not in_position:
            position = capital / price
            entry_price = price
            capital = 0.0
            in_position = True
            trades.append({"date": date, "type": "BUY", "price": price, "position": position})

        elif signals.iloc[i] == -1 and in_position:
            capital = position * price
            pnl_pct = (price - entry_price) / entry_price * 100
            trades.append({"date": date, "type": "SELL", "price": price, "pnl_pct": pnl_pct, "capital": capital})
            position = 0.0
            in_position = False

        eq = capital + position * price if in_position else capital
        equity.append(eq)

    equity = pd.Series(equity, index=df.index)
    final_val = equity.iloc[-1]
    total_return = (final_val / initial_capital - 1) * 100
    buy_hold_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    # Metrics
    trade_df = pd.DataFrame(trades)
    sells = trade_df[trade_df["type"] == "SELL"] if len(trade_df) > 0 else pd.DataFrame()
    num_trades = len(sells)
    win_rate = (sells["pnl_pct"] > 0).mean() * 100 if num_trades > 0 else 0
    avg_trade = sells["pnl_pct"].mean() if num_trades > 0 else 0
    best_trade = sells["pnl_pct"].max() if num_trades > 0 else 0
    worst_trade = sells["pnl_pct"].min() if num_trades > 0 else 0

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min() * 100

    # Sharpe (annualized)
    daily_ret = equity.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() > 0 else 0

    # Profit factor
    if num_trades > 0:
        gross_profit = sells[sells["pnl_pct"] > 0]["pnl_pct"].sum()
        gross_loss = abs(sells[sells["pnl_pct"] < 0]["pnl_pct"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        profit_factor = 0

    results = {
        "name": name,
        "total_return_pct": round(total_return, 2),
        "buy_hold_return_pct": round(buy_hold_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 3),
        "num_trades": num_trades,
        "win_rate_pct": round(win_rate, 1),
        "avg_trade_pct": round(avg_trade, 2),
        "best_trade_pct": round(best_trade, 2),
        "worst_trade_pct": round(worst_trade, 2),
        "profit_factor": round(profit_factor, 2),
        "final_value": round(final_val, 2),
    }
    return results, equity, trades
