#!/usr/bin/env python3
"""Run all strategy versions and rank them."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.engine import load_data, backtest
from backtest.strategies import ALL_STRATEGIES as STRATS_V1
from backtest.strategies_v2 import ALL_STRATEGIES as STRATS_V2
import pandas as pd

def main():
    df = load_data("backtest/btc_daily.csv")
    print(f"📊 BTC/USD Daily Backtest | {df.index[0].date()} to {df.index[-1].date()} | {len(df)} bars")
    print(f"   Start: ${df['close'].iloc[0]:,.0f} | End: ${df['close'].iloc[-1]:,.0f}")
    bh = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    print(f"   Buy & Hold: {bh:.1f}%\n")

    all_results = []
    all_strats = {}
    all_strats.update({"[R1] " + k: v for k, v in STRATS_V1.items()})
    all_strats.update({"[R2] " + k: v for k, v in STRATS_V2.items()})

    for name, fn in all_strats.items():
        try:
            results, equity, trades = backtest(df, fn, initial_capital=10000, name=name)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR in {name}: {e}")

    results_df = pd.DataFrame(all_results).sort_values("total_return_pct", ascending=False)

    print(f"{'#':<3} {'Strategy':<40} {'Return%':>10} {'MaxDD%':>10} {'Sharpe':>8} {'Trades':>7} {'WinR%':>7} {'AvgTr%':>8} {'PF':>7} {'Final$':>12}")
    print("=" * 115)
    for i, (_, r) in enumerate(results_df.iterrows(), 1):
        pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "  inf"
        print(f"{i:<3} {r['name']:<40} {r['total_return_pct']:>9.1f}% {r['max_drawdown_pct']:>9.1f}% {r['sharpe_ratio']:>8.3f} {r['num_trades']:>7} {r['win_rate_pct']:>6.1f}% {r['avg_trade_pct']:>7.1f}% {pf:>7} ${r['final_value']:>10,.0f}")
    print("=" * 115)
    print(f"    {'Buy & Hold Benchmark':<40} {bh:>9.1f}%")

    # Top 3
    print("\n" + "=" * 60)
    print("TOP 3 STRATEGIES")
    print("=" * 60)
    for i, (_, r) in enumerate(results_df.head(3).iterrows(), 1):
        beat = "BEATS" if r['total_return_pct'] > bh else "BELOW"
        print(f"\n  #{i} {r['name']}")
        print(f"     Return: {r['total_return_pct']:.1f}% ({beat} B&H)")
        print(f"     Sharpe: {r['sharpe_ratio']:.3f} | Max DD: {r['max_drawdown_pct']:.1f}% | Win Rate: {r['win_rate_pct']:.1f}%")
        print(f"     Trades: {r['num_trades']} | Avg Trade: {r['avg_trade_pct']:.1f}% | Final: ${r['final_value']:,.0f}")

if __name__ == "__main__":
    main()
