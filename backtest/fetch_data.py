#!/usr/bin/env python3
"""Fetch BTC/USD daily data from CoinGecko free API."""
import urllib.request
import json
import csv
import os
from datetime import datetime

def fetch_btc_data():
    """Fetch max BTC/USD daily data from CoinGecko."""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=max&interval=daily"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    prices = data["prices"]
    volumes = data.get("total_volumes", [])

    vol_map = {}
    for ts, vol in volumes:
        d = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        vol_map[d] = vol

    out_path = os.path.join(os.path.dirname(__file__), "btc_daily.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "close", "volume"])
        for ts, price in prices:
            d = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
            w.writerow([d, round(price, 2), round(vol_map.get(d, 0), 0)])

    print(f"Saved {len(prices)} daily candles to {out_path}")
    return out_path

if __name__ == "__main__":
    fetch_btc_data()
