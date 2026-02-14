#!/usr/bin/env python3
"""Run all strategies and compare results."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.engine import load_data, backtest
from backtest.strategies import ALL_STRATEGIES
import pandas as pd

def main():
    df = load_data("backtest/btc_daily.csv")
    print(f"Data: {df.index[0].date()} to {df.index[-1].date()} | {len(df)} bars")
    print(f"Start: ${df['close'].iloc[0]:,.0f} | End: ${df['close'].iloc[-1]:,.0f}")
    print(f"Buy & Hold Return: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.1f}%")
    print("=" * 110)

    all_results = []
    for name, fn in ALL_STRATEGIES.items():
        results, equity, trades = backtest(df, fn, initial_capital=10000, name=name)
        all_results.append(results)

    # Sort by total return
    results_df = pd.DataFrame(all_results).sort_values("total_return_pct", ascending=False)

    print(f"\n{'Strategy':<35} {'Return%':>10} {'MaxDD%':>10} {'Sharpe':>8} {'Trades':>8} {'WinR%':>8} {'AvgTr%':>8} {'PF':>8} {'Final$':>12}")
    print("-" * 110)
    for _, r in results_df.iterrows():
        print(f"{r['name']:<35} {r['total_return_pct']:>9.1f}% {r['max_drawdown_pct']:>9.1f}% {r['sharpe_ratio']:>8.3f} {r['num_trades']:>8} {r['win_rate_pct']:>7.1f}% {r['avg_trade_pct']:>7.1f}% {r['profit_factor']:>7.2f} ${r['final_value']:>10,.0f}")

    print("-" * 110)
    bh = results_df.iloc[0]["buy_hold_return_pct"]
    print(f"{'Buy & Hold Benchmark':<35} {bh:>9.1f}%")

    # Identify best
    best = results_df.iloc[0]
    print(f"\n🏆 BEST STRATEGY: {best['name']}")
    print(f"   Return: {best['total_return_pct']:.1f}% | Sharpe: {best['sharpe_ratio']:.3f} | Max DD: {best['max_drawdown_pct']:.1f}% | Win Rate: {best['win_rate_pct']:.1f}%")

    # Best risk-adjusted
    best_sharpe = results_df.loc[results_df["sharpe_ratio"].idxmax()]
    print(f"\n⭐ BEST RISK-ADJUSTED: {best_sharpe['name']}")
    print(f"   Return: {best_sharpe['total_return_pct']:.1f}% | Sharpe: {best_sharpe['sharpe_ratio']:.3f} | Max DD: {best_sharpe['max_drawdown_pct']:.1f}%")

if __name__ == "__main__":
    main()
