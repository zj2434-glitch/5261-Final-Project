import os
import pandas as pd
import numpy as np
from yahooquery import Ticker

#Tech stocks across large-, mid-, and small-cap groups
tickers = [
    # Large-cap
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL",
    # Mid-cap
    "ENPH", "ESTC", "VRRM", "LSCC", "QRVO",
    # Small-cap
    "ACMR", "APPN", "PD", "VERI", "INOD"]

start = "2016-01-01"
end = "2025-10-01"

#Download data from Yahoo Finance
yq = Ticker(tickers)
data = yq.history(start=start, end=end, interval="1d", adj_ohlc=True).reset_index()

# Keep only useful columns
possible_cols = [
    "symbol", "date", "open", "high", "low", "close",
    "adjclose", "adjopen", "adjhigh", "adjlow", "volume"
]
keep = [c for c in possible_cols if c in data.columns]
data = data[keep].copy()

# Process date column
data["date"] = pd.to_datetime(data["date"])
data = data[(data["date"] >= start) & (data["date"] <= end)].sort_values(["symbol", "date"])

#Output folder
out_dir = "data_tech_2016_2025"
os.makedirs(out_dir, exist_ok=True)
date_tag = f"{start.replace('-', '')}_{end.replace('-', '')}"


for sym in tickers:
    df = data[data["symbol"] == sym].copy()
    if df.empty:
        print(f"{sym} — no data available")
        continue
    # If adjclose is missing, use close instead
    if "adjclose" not in df.columns:
        df["adjclose"] = df["close"]
    # Calculate daily returns
    df["ret_d1"] = df["adjclose"].pct_change()
    df["log_ret_d1"] = np.log1p(df["ret_d1"])

    # Save to CSV
    df = df.sort_values("date").reset_index(drop=True)
    out_path = os.path.join(out_dir, f"{sym}_{date_tag}_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"{sym} saved → {out_path}")
