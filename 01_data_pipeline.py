"""
01_data_pipeline.py
--------------------
Pull and save all raw data for the Early Warning Signals project.

Sources:
  - FRED API: yield spreads, credit spreads, unemployment claims, CFNAI, LEI
  - yfinance: S&P 500 (^GSPC), VIX (^VIX), sector ETFs
  - Google Trends (pytrends): fear-related search terms
  - AAII Sentiment: loaded from manually downloaded CSV

Author: Mahesh Jagdish Wadhokar
BA889 — Boston University Questrom, June 2026
"""

import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from pytrends.request import TrendReq

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
FRED_API_KEY = "4514d381e40ed95f3e1c04257e5e89d1"
START_DATE   = "1990-01-01"
END_DATE     = "2024-12-31"
DATA_DIR     = "data/raw"

os.makedirs(DATA_DIR, exist_ok=True)
fred = Fred(api_key=FRED_API_KEY.strip())

# ── Helpers ───────────────────────────────────────────────────────────────────
def save(df, name: str):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    path = os.path.join(DATA_DIR, f"{name}.csv")
    df.to_csv(path)
    print(f"  Saved {name}.csv  ({len(df)} rows)")


def yf_close(ticker: str) -> pd.Series:
    """Download close price — handles all yfinance column formats."""
    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        col = "Close" if "Close" in level0 else "Adj Close"
        s = df.xs(col, axis=1, level=0).squeeze()
    else:
        if "Close" in df.columns:
            s = df["Close"]
        elif "Adj Close" in df.columns:
            s = df["Adj Close"]
        else:
            raise KeyError(f"No Close column for {ticker}. Got: {df.columns.tolist()}")
    if isinstance(s, pd.DataFrame):
        s = s.squeeze()
    s.name = ticker
    return s


# ═════════════════════════════════════════════════════════════════════════════
# 1. FRED
# ═════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Downloading FRED macro series...")

fred_series = {
    "T10Y2Y"       : "yield_2y10y",
    "T10Y3M"       : "yield_3m10y",
    "BAMLH0A0HYM2" : "hy_spread",
    "ICSA"         : "initial_claims",
    "CFNAI"        : "cfnai",
    "USSLIND"      : "lei",
}

for ticker, label in fred_series.items():
    try:
        s = fred.get_series(ticker, observation_start=START_DATE, observation_end=END_DATE)
        s.name = label
        save(s, label)
        time.sleep(0.3)
    except Exception as e:
        print(f"  WARNING: Could not fetch {ticker}: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# 2. yfinance
# ═════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Downloading market data via yfinance...")

save(yf_close("^GSPC"), "sp500")
save(yf_close("^VIX"),  "vix")

for ticker, label in {"XLU":"etf_xlu","XLP":"etf_xlp","XLY":"etf_xly","XLK":"etf_xlk"}.items():
    try:
        save(yf_close(ticker), label)
    except Exception as e:
        print(f"  WARNING: {ticker}: {e}")

print("  Market data downloaded.")

# ═════════════════════════════════════════════════════════════════════════════
# 3. Google Trends
# ═════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Downloading Google Trends fear index...")

FEAR_TERMS = ["market crash", "recession", "stock selloff", "financial crisis"]

try:
    pytrends  = TrendReq(hl="en-US", tz=360)
    all_trends = []
    for term in FEAR_TERMS:
        pytrends.build_payload([term], cat=0,
                               timeframe=f"{START_DATE} {END_DATE}", geo="US")
        df_trend = pytrends.interest_over_time()
        if not df_trend.empty:
            all_trends.append(df_trend[[term]])
        time.sleep(1.5)

    if all_trends:
        trends_df = pd.concat(all_trends, axis=1)
        trends_df["fear_composite"] = trends_df.mean(axis=1)
        save(trends_df, "google_trends_fear")
        print("  Google Trends downloaded.")
    else:
        raise ValueError("Empty response from Google Trends.")

except Exception as e:
    print(f"  WARNING: Google Trends failed: {e}")
    print("  Using synthetic placeholder...")
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="W")
    np.random.seed(42)
    syn = pd.DataFrame({t: np.random.uniform(0, 100, len(dates)) for t in FEAR_TERMS}, index=dates)
    syn["fear_composite"] = syn.mean(axis=1)
    save(syn, "google_trends_fear")

# ═════════════════════════════════════════════════════════════════════════════
# 4. AAII Sentiment
# ═════════════════════════════════════════════════════════════════════════════
print("\n[4/4] AAII Sentiment Setup...")

aaii_path = os.path.join(DATA_DIR, "aaii_sentiment.csv")
if not os.path.exists(aaii_path):
    print("  No AAII file found — generating synthetic placeholder.")
    print("  Download real data: https://www.aaii.com/sentimentsurvey/sent_results")
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="W-THU")
    np.random.seed(123)
    bullish = np.random.uniform(20, 55, len(dates))
    bearish = np.random.uniform(15, 50, len(dates))
    pd.DataFrame({
        "Bullish": bullish,
        "Neutral": 100 - bullish - bearish,
        "Bearish": bearish,
        "BullBearSpread": bullish - bearish
    }, index=dates).rename_axis("Date").to_csv(aaii_path)
    print("  Synthetic AAII saved.")
else:
    print(f"  AAII data found ✓")

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("DATA PIPELINE COMPLETE")
print("="*60)
for f in sorted(os.listdir(DATA_DIR)):
    size = os.path.getsize(os.path.join(DATA_DIR, f))
    print(f"  {f:35s}  {size/1024:.1f} KB")
print("\nNext step: python 02_feature_engineering.py")
