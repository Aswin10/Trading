#!/usr/bin/env python3
"""Generate realistic BTC/USD daily data mimicking historical cycles."""
import numpy as np
import pandas as pd
import os

def generate_btc_data():
    np.random.seed(42)
    days = 365 * 8  # ~8 years (2017-2024)
    dates = pd.date_range("2017-01-01", periods=days, freq="D")

    # BTC-like cycles: ~4 year halving cycles with blow-off tops and 80% drawdowns
    t = np.arange(days)

    # Long-term uptrend (log growth)
    trend = 900 * np.exp(0.0012 * t)

    # 4-year cycle (~1460 days)
    cycle_4y = 0.6 * np.sin(2 * np.pi * t / 1460 - np.pi / 2)

    # 1-year seasonal cycle
    cycle_1y = 0.15 * np.sin(2 * np.pi * t / 365 - np.pi / 3)

    # Shorter swing cycle (~90 days)
    cycle_90d = 0.08 * np.sin(2 * np.pi * t / 90)

    # Combine cycles with asymmetric blow-off tops
    combined = cycle_4y + cycle_1y + cycle_90d
    # Make tops sharper, bottoms rounder (asymmetry)
    combined = np.where(combined > 0, combined * 1.5, combined * 0.8)

    # Random walk component
    noise = np.cumsum(np.random.normal(0, 0.02, days))
    daily_noise = np.random.normal(0, 0.03, days)

    log_price = np.log(trend) + combined + noise * 0.3 + daily_noise
    close = np.exp(log_price)

    # Generate OHLC from close
    daily_range = np.random.uniform(0.01, 0.06, days)
    high = close * (1 + daily_range / 2)
    low = close * (1 - daily_range / 2)
    open_shift = np.random.uniform(-0.02, 0.02, days)
    open_ = close * (1 + open_shift)
    open_ = np.clip(open_, low, high)

    volume = np.random.lognormal(mean=20, sigma=1.5, size=days)
    # Volume spikes at extremes
    vol_mult = 1 + 2 * np.abs(combined)
    volume = volume * vol_mult

    df = pd.DataFrame({
        "date": dates,
        "open": np.round(open_, 2),
        "high": np.round(high, 2),
        "low": np.round(low, 2),
        "close": np.round(close, 2),
        "volume": np.round(volume, 0),
    })

    out_path = os.path.join(os.path.dirname(__file__), "btc_daily.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} daily candles from {dates[0].date()} to {dates[-1].date()}")
    print(f"Price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")
    print(f"Saved to {out_path}")
    return out_path

if __name__ == "__main__":
    generate_btc_data()
