#!/usr/bin/env python3
"""Optimized BTC/USD cycle trading strategies — tuned for crypto volatility."""
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

def stoch_rsi(series, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    r = rsi(series, rsi_period)
    lowest = r.rolling(stoch_period).min()
    highest = r.rolling(stoch_period).max()
    stoch_k = 100 * (r - lowest) / (highest - lowest + 1e-10)
    stoch_d = sma(stoch_k, d_period)
    return stoch_k, stoch_d

def atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ──────────────────────────────────────────────────────
# V1: RSI Cycle Swings (tuned for BTC)
# Buy: RSI(21) < 40 AND price > EMA100 (buy dips in uptrend)
# Sell: RSI(21) > 65 (take profit on strength)
# ──────────────────────────────────────────────────────
def strategy_v1_rsi_swings(df):
    r = rsi(df["close"], 21)
    ma = ema(df["close"], 100)
    signals = pd.Series(0, index=df.index)
    signals[(r < 40) & (df["close"] > ma)] = 1
    signals[r > 65] = -1
    return signals


# ──────────────────────────────────────────────────────
# V2: Golden Cross Trend Follower
# Buy: EMA21 > EMA55 (trend confirmed)
# Sell: EMA21 < EMA55 (trend lost)
# Faster MAs than V1 for more trades
# ──────────────────────────────────────────────────────
def strategy_v2_fast_ma_cross(df):
    ema21 = ema(df["close"], 21)
    ema55 = ema(df["close"], 55)
    signals = pd.Series(0, index=df.index)
    cross_up = (ema21 > ema55) & (ema21.shift(1) <= ema55.shift(1))
    cross_dn = (ema21 < ema55) & (ema21.shift(1) >= ema55.shift(1))
    signals[cross_up] = 1
    signals[cross_dn] = -1
    return signals


# ──────────────────────────────────────────────────────
# V3: Stochastic RSI + EMA Trend
# Buy: StochRSI K crosses above D from oversold (<20) + price > EMA50
# Sell: StochRSI K crosses below D from overbought (>80)
# ──────────────────────────────────────────────────────
def strategy_v3_stoch_rsi(df):
    k, d = stoch_rsi(df["close"])
    ma = ema(df["close"], 50)
    signals = pd.Series(0, index=df.index)

    buy = (k > d) & (k.shift(1) <= d.shift(1)) & (k < 30) & (df["close"] > ma)
    sell = (k < d) & (k.shift(1) >= d.shift(1)) & (k > 70)
    signals[buy] = 1
    signals[sell] = -1
    return signals


# ──────────────────────────────────────────────────────
# V4: Bollinger Band Mean Reversion
# Buy: Close below lower BB(20,2) for 2 consecutive days, then closes above it
# Sell: Close above upper BB or after 30% gain
# ──────────────────────────────────────────────────────
def strategy_v4_bb_reversion(df):
    mid, upper, lower = bb(df["close"], 20, 2.0)
    signals = pd.Series(0, index=df.index)

    was_below = (df["close"].shift(1) < lower.shift(1))
    bounce = df["close"] > lower
    signals[was_below & bounce] = 1
    signals[df["close"] > upper] = -1
    return signals


# ──────────────────────────────────────────────────────
# V5: MACD + ATR Volatility Filter
# Buy: MACD histogram crosses positive + ATR expanding (volatility breakout)
# Sell: MACD histogram crosses negative
# ──────────────────────────────────────────────────────
def strategy_v5_macd_atr(df):
    _, _, hist = macd(df["close"], 12, 26, 9)
    a = atr(df, 14)
    a_sma = sma(a, 20)
    signals = pd.Series(0, index=df.index)

    macd_cross_up = (hist > 0) & (hist.shift(1) <= 0)
    vol_expanding = a > a_sma
    signals[macd_cross_up & vol_expanding] = 1

    macd_cross_dn = (hist < 0) & (hist.shift(1) >= 0)
    signals[macd_cross_dn] = -1
    return signals


# ──────────────────────────────────────────────────────
# V6: Cycle Composite — Combines 3 indicators
# Buy: RSI(14) < 45 + MACD hist > 0 + Price > SMA50 (trifecta buy)
# Sell: RSI(14) > 70 OR Price < SMA50 * 0.93 (dynamic stop)
# ──────────────────────────────────────────────────────
def strategy_v6_composite(df):
    r = rsi(df["close"], 14)
    _, _, hist = macd(df["close"])
    ma50 = sma(df["close"], 50)
    signals = pd.Series(0, index=df.index)

    buy = (r < 45) & (hist > 0) & (df["close"] > ma50)
    signals[buy] = 1

    sell = (r > 70) | (df["close"] < ma50 * 0.93)
    signals[sell] = -1
    return signals


# ──────────────────────────────────────────────────────
# V7: Trend Following with Trailing Stop
# Buy: Price > EMA50 AND EMA50 > EMA200 AND RSI > 50
# Sell: Price drops 15% from recent high OR EMA50 < EMA200
# ──────────────────────────────────────────────────────
def strategy_v7_trend_trailing(df):
    ema50 = ema(df["close"], 50)
    ema200 = ema(df["close"], 200)
    r = rsi(df["close"], 14)
    signals = pd.Series(0, index=df.index)

    uptrend = (ema50 > ema200) & (df["close"] > ema50) & (r > 50)
    prev_not = ~((ema50.shift(1) > ema200.shift(1)) & (df["close"].shift(1) > ema50.shift(1)))
    signals[uptrend & prev_not] = 1

    # Rolling 30-day high for trailing stop
    high_30 = df["close"].rolling(30).max()
    trail_stop = df["close"] < high_30 * 0.85
    death_cross = (ema50 < ema200) & (ema50.shift(1) >= ema200.shift(1))
    signals[trail_stop | death_cross] = -1
    return signals


# ──────────────────────────────────────────────────────
# V8: Aggressive Swing Trader
# Buy: 5-day RSI < 25 (short-term oversold bounce)
# Sell: 5-day RSI > 75 (short-term overbought)
# ──────────────────────────────────────────────────────
def strategy_v8_aggressive_swing(df):
    r5 = rsi(df["close"], 5)
    signals = pd.Series(0, index=df.index)
    signals[r5 < 25] = 1
    signals[r5 > 75] = -1
    return signals


ALL_STRATEGIES = {
    "V1: RSI Cycle Swings": strategy_v1_rsi_swings,
    "V2: Fast MA Cross (21/55)": strategy_v2_fast_ma_cross,
    "V3: Stoch RSI + Trend": strategy_v3_stoch_rsi,
    "V4: BB Mean Reversion": strategy_v4_bb_reversion,
    "V5: MACD + ATR Breakout": strategy_v5_macd_atr,
    "V6: Cycle Composite": strategy_v6_composite,
    "V7: Trend + Trailing Stop": strategy_v7_trend_trailing,
    "V8: Aggressive Swing (RSI5)": strategy_v8_aggressive_swing,
}
