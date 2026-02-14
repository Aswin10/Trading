#!/usr/bin/env python3
"""Six BTC/USD cycle trading strategies."""
import pandas as pd
import numpy as np


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def sma(series, period):
    return series.rolling(period).mean()


def bb(series, period=20, std_mult=2.0):
    mid = sma(series, period)
    std = series.rolling(period).std()
    return mid, mid + std_mult * std, mid - std_mult * std


def macd(series, fast=12, slow=26, signal=9):
    f = ema(series, fast)
    s = ema(series, slow)
    macd_line = f - s
    sig = ema(macd_line, signal)
    hist = macd_line - sig
    return macd_line, sig, hist


# ──────────────────────────────────────────────────────
# V1: RSI Mean Reversion + 200 SMA Trend Filter
# Buy: RSI < 30 and price > SMA200 (uptrend dips)
# Sell: RSI > 70
# ──────────────────────────────────────────────────────
def strategy_v1_rsi_mean_reversion(df):
    r = rsi(df["close"], 14)
    ma200 = sma(df["close"], 200)
    signals = pd.Series(0, index=df.index)
    signals[(r < 30) & (df["close"] > ma200)] = 1
    signals[r > 70] = -1
    return signals


# ──────────────────────────────────────────────────────
# V2: Dual MA Crossover + RSI Filter
# Buy: EMA50 crosses above EMA200 (golden cross) + RSI < 60
# Sell: EMA50 crosses below EMA200 (death cross)
# ──────────────────────────────────────────────────────
def strategy_v2_dual_ma_crossover(df):
    ema50 = ema(df["close"], 50)
    ema200 = ema(df["close"], 200)
    r = rsi(df["close"], 14)
    signals = pd.Series(0, index=df.index)

    prev_above = (ema50.shift(1) > ema200.shift(1))
    curr_above = (ema50 > ema200)
    golden = ~prev_above & curr_above
    death = prev_above & ~curr_above

    signals[golden & (r < 60)] = 1
    signals[death] = -1
    return signals


# ──────────────────────────────────────────────────────
# V3: Bollinger Band Squeeze + MACD Confirmation
# Buy: Price touches lower BB + MACD histogram turns positive
# Sell: Price touches upper BB or MACD histogram turns negative after being positive
# ──────────────────────────────────────────────────────
def strategy_v3_bb_macd(df):
    mid, upper, lower = bb(df["close"], 20, 2.0)
    _, _, hist = macd(df["close"])
    signals = pd.Series(0, index=df.index)

    near_lower = df["close"] <= lower * 1.02
    macd_bull = (hist > 0) & (hist.shift(1) <= 0)
    signals[near_lower & macd_bull] = 1

    near_upper = df["close"] >= upper * 0.98
    macd_bear = (hist < 0) & (hist.shift(1) >= 0)
    signals[near_upper | macd_bear] = -1
    return signals


# ──────────────────────────────────────────────────────
# V4: Multi-Timeframe Momentum (Weekly + Daily)
# Simulates weekly with 50-day MA as trend, daily RSI for entry
# Buy: Price > SMA50, RSI crosses above 40 from below (momentum resumption)
# Sell: RSI > 75 or Price < SMA50 * 0.95 (trailing stop)
# ──────────────────────────────────────────────────────
def strategy_v4_multi_tf_momentum(df):
    ma50 = sma(df["close"], 50)
    r = rsi(df["close"], 14)
    signals = pd.Series(0, index=df.index)

    uptrend = df["close"] > ma50
    rsi_cross_up = (r > 40) & (r.shift(1) <= 40)
    signals[uptrend & rsi_cross_up] = 1

    overbought = r > 75
    stop_hit = df["close"] < ma50 * 0.95
    signals[overbought | stop_hit] = -1
    return signals


# ──────────────────────────────────────────────────────
# V5: Accumulation Zone (200 SMA Discount + RSI + Volume)
# Buy: Price < SMA200 * 0.9 (deep discount) + RSI < 35 + volume spike
# Sell: Price > SMA200 * 1.3 (30% above 200MA) or RSI > 80
# Designed for catching cycle bottoms
# ──────────────────────────────────────────────────────
def strategy_v5_accumulation_zone(df):
    ma200 = sma(df["close"], 200)
    r = rsi(df["close"], 21)
    vol_ma = sma(df["volume"], 50)
    signals = pd.Series(0, index=df.index)

    deep_discount = df["close"] < ma200 * 0.90
    oversold = r < 35
    vol_spike = df["volume"] > vol_ma * 1.2
    signals[deep_discount & oversold & vol_spike] = 1

    premium = df["close"] > ma200 * 1.30
    overbought = r > 80
    signals[premium | overbought] = -1
    return signals


# ──────────────────────────────────────────────────────
# V6: Adaptive Cycle — PI Cycle inspired + RSI Divergence
# Uses 111 SMA and 350 SMA/2 (PI cycle top indicator) for tops
# Uses 200 SMA and RSI < 30 for bottoms
# Buy: Price near SMA200, RSI < 35, price starts rising (3-day up)
# Sell: 111 SMA crosses above 350 SMA / 2 (PI cycle top) or RSI > 80
# ──────────────────────────────────────────────────────
def strategy_v6_pi_cycle(df):
    ma111 = sma(df["close"], 111)
    ma350_2 = sma(df["close"], 350) / 2
    ma200 = sma(df["close"], 200)
    r = rsi(df["close"], 14)
    signals = pd.Series(0, index=df.index)

    near_200 = df["close"] < ma200 * 1.05
    oversold = r < 35
    rising = (df["close"] > df["close"].shift(1)) & (df["close"].shift(1) > df["close"].shift(2))
    signals[near_200 & oversold & rising] = 1

    pi_top = (ma111 > ma350_2) & (ma111.shift(1) <= ma350_2.shift(1))
    extreme_overbought = r > 80
    signals[pi_top | extreme_overbought] = -1
    return signals


ALL_STRATEGIES = {
    "V1: RSI Mean Reversion + 200SMA": strategy_v1_rsi_mean_reversion,
    "V2: Dual MA Crossover + RSI": strategy_v2_dual_ma_crossover,
    "V3: BB Squeeze + MACD": strategy_v3_bb_macd,
    "V4: Multi-TF Momentum": strategy_v4_multi_tf_momentum,
    "V5: Accumulation Zone": strategy_v5_accumulation_zone,
    "V6: PI Cycle Adaptive": strategy_v6_pi_cycle,
}
