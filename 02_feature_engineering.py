"""
02_feature_engineering.py
--------------------------
Merges all raw data sources, engineers 12 predictive features,
and constructs forward-looking binary crash labels.

Author: Mahesh Jagdish Wadhokar
BA889 — Boston University Questrom, June 2026
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

RAW_DIR  = "data/raw"
PROC_DIR = "data/processed"
FIG_DIR  = "reports/figures"
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Load raw data
# ═════════════════════════════════════════════════════════════════════════════
print("[1/5] Loading raw data...")

def load_csv(name, index_col=0, parse_dates=True):
    path = os.path.join(RAW_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}. Run 01_data_pipeline.py first.")
    return pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)

sp500   = load_csv("sp500").squeeze()
vix     = load_csv("vix").squeeze()
y2y10   = load_csv("yield_2y10y").squeeze()
y3m10   = load_csv("yield_3m10y").squeeze()
hy      = load_csv("hy_spread").squeeze()
claims  = load_csv("initial_claims").squeeze()
cfnai   = load_csv("cfnai").squeeze()
lei     = load_csv("lei").squeeze()
trends  = load_csv("google_trends_fear")
aaii    = load_csv("aaii_sentiment")

etf_xlu = load_csv("etf_xlu").squeeze()
etf_xlp = load_csv("etf_xlp").squeeze()
etf_xly = load_csv("etf_xly").squeeze()
etf_xlk = load_csv("etf_xlk").squeeze()

print("  All raw files loaded ✓")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Construct crash labels
# ═════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Constructing forward-looking crash labels...")

def label_drawdown(prices: pd.Series, threshold: float = -0.15, window: int = 126) -> pd.Series:
    labels     = []
    prices_arr = prices.values
    for i in range(len(prices_arr)):
        fwd_end    = min(i + window, len(prices_arr))
        fwd_prices = prices_arr[i:fwd_end]
        if len(fwd_prices) < 2:
            labels.append(np.nan)
            continue
        rolling_max = np.maximum.accumulate(fwd_prices)
        drawdowns   = (fwd_prices - rolling_max) / rolling_max
        max_dd      = drawdowns.min()
        labels.append(1 if max_dd <= threshold else 0)
    return pd.Series(labels, index=prices.index, name="crash_label")

crash_labels = label_drawdown(sp500)
crash_rate   = crash_labels.mean() * 100
print(f"  Crash label rate: {crash_rate:.1f}% of days labeled as pre-crash")
print(f"  Class imbalance ratio: 1:{(1-crash_labels.mean())/crash_labels.mean():.1f}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: Engineer features
# ═════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Engineering features...")

def to_monthly(s: pd.Series, method="last") -> pd.Series:
    s = s.dropna()
    return s.resample("ME").last() if method == "last" else s.resample("ME").mean()

sp500_m  = to_monthly(sp500)
vix_m    = to_monthly(vix, method="mean")
y2y10_m  = to_monthly(y2y10)
y3m10_m  = to_monthly(y3m10)
hy_m     = to_monthly(hy)
labels_m = crash_labels.resample("ME").last()

claims_m = claims.resample("ME").mean()
aaii_m   = aaii.resample("ME").last()

trends_m = trends["fear_composite"].resample("ME").mean() if "fear_composite" in trends.columns \
           else trends.iloc[:, 0].resample("ME").mean()

cfnai_m = cfnai.resample("ME").last()
lei_m   = lei.resample("ME").last()

xlu_m = to_monthly(etf_xlu)
xlp_m = to_monthly(etf_xlp)
xly_m = to_monthly(etf_xly)
xlk_m = to_monthly(etf_xlk)

features = pd.DataFrame(index=sp500_m.index)
features["yield_2y10y"]    = y2y10_m
features["yield_3m10y"]    = y3m10_m
features["hy_spread"]      = hy_m
features["hy_spread_vel"]  = hy_m.diff(3)
features["vix_level"]      = vix_m
features["vix_change_30d"] = vix_m.pct_change(1) * 100

if "BullBearSpread" in aaii_m.columns:
    features["aaii_bull_bear"] = aaii_m["BullBearSpread"]
elif "Bullish" in aaii_m.columns and "Bearish" in aaii_m.columns:
    features["aaii_bull_bear"] = aaii_m["Bullish"] - aaii_m["Bearish"]
else:
    features["aaii_bull_bear"] = aaii_m.iloc[:, 0]

features["trends_fear"]    = trends_m
claims_ma4 = claims.rolling(4).mean()
features["claims_ma4_chg"] = claims_ma4.pct_change(1).resample("ME").last() * 100
features["cfnai_ma3"]      = cfnai_m.rolling(3).mean()
features["lei_mom"]        = lei_m.pct_change(1) * 100

defensive_ret = ((xlu_m + xlp_m) / 2).pct_change(3)
cyclical_ret  = ((xly_m + xlk_m) / 2).pct_change(3)
features["sector_rotation"] = (defensive_ret - cyclical_ret) * 100

print(f"  Features engineered: {features.shape[1]}")
for col in features.columns:
    pct_missing = features[col].isna().mean() * 100
    print(f"    {col:25s}  {pct_missing:.1f}% missing")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: Merge and clean
# ═════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Merging and cleaning...")

dataset = features.copy()
dataset["crash_label"] = labels_m
dataset = dataset.loc["1995-01-01":"2024-12-31"]
dataset = dataset.dropna(subset=["crash_label"])
dataset["crash_label"] = dataset["crash_label"].astype(int)

# Replace inf/-inf before filling
dataset = dataset.replace([np.inf, -np.inf], np.nan)

# Forward-fill then backfill
dataset = dataset.ffill().bfill()

# Final inf/nan sweep
dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.dropna()

rows_after = len(dataset)
print(f"  Rows after cleaning: {rows_after}")
print(f"  Date range: {dataset.index.min().date()} → {dataset.index.max().date()}")
print(f"  Crash label distribution: {dict(dataset['crash_label'].value_counts().sort_index())}")

out_path = os.path.join(PROC_DIR, "dataset_features_labels.csv")
dataset.to_csv(out_path)
print(f"\n  Saved: {out_path}")

FEATURE_COLS = [c for c in dataset.columns if c != "crash_label"]


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: EDA visualizations
# ═════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Generating EDA plots...")

# Plot 1: Crash label overlay
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle("Early Warning Signals — Data Overview", fontsize=14, fontweight="bold")

sp500_plot = sp500_m.loc[dataset.index]
crash_mask = dataset["crash_label"] == 1

axes[0].plot(sp500_plot.index, sp500_plot.values, color="#1a1a2e", linewidth=1, label="S&P 500")
axes[0].fill_between(sp500_plot.index, sp500_plot.min(), sp500_plot.max(),
                     where=crash_mask.reindex(sp500_plot.index, fill_value=False),
                     alpha=0.25, color="#e74c3c", label="Pre-crash zone (label=1)")
axes[0].set_ylabel("S&P 500 Price")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

axes[1].plot(dataset.index, dataset["yield_2y10y"], color="#2980b9", linewidth=1.2, label="10yr-2yr spread")
axes[1].plot(dataset.index, dataset["yield_3m10y"], color="#8e44ad", linewidth=1.2, alpha=0.7, label="10yr-3mo spread")
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].fill_between(dataset.index, dataset["yield_2y10y"].min(), dataset["yield_2y10y"].max(),
                     where=crash_mask, alpha=0.15, color="#e74c3c")
axes[1].set_ylabel("Yield Spread (%)")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

axes[2].plot(dataset.index, dataset["vix_level"], color="#e67e22", linewidth=1.2, label="VIX")
axes[2].plot(dataset.index, dataset["hy_spread"], color="#c0392b", linewidth=1.2, alpha=0.8, label="HY Spread")
axes[2].fill_between(dataset.index, 0, dataset[["vix_level","hy_spread"]].max().max(),
                     where=crash_mask, alpha=0.15, color="#e74c3c")
axes[2].set_ylabel("Level")
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eda_overview.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR}/eda_overview.png")

# Plot 2: Correlation heatmap
import seaborn as sns
fig, ax = plt.subplots(figsize=(11, 9))
corr = dataset[FEATURE_COLS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
            annot_kws={"size": 7}, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "feature_correlation.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR}/feature_correlation.png")

# Plot 3: Feature distributions — clip to finite values before plotting
fig, axes = plt.subplots(3, 4, figsize=(16, 11))
axes = axes.flatten()
for i, feat in enumerate(FEATURE_COLS[:12]):
    ax  = axes[i]
    col = dataset[feat].replace([np.inf, -np.inf], np.nan).dropna()

    # Clip to 1st–99th percentile to avoid plotting extreme outliers
    lo, hi = col.quantile(0.01), col.quantile(0.99)
    col_clipped = col.clip(lo, hi)

    d0 = col_clipped[dataset.loc[col.index, "crash_label"] == 0]
    d1 = col_clipped[dataset.loc[col.index, "crash_label"] == 1]

    if len(d0) > 1:
        ax.hist(d0, bins=25, alpha=0.6, color="#3498db", label="No crash (0)", density=True)
    if len(d1) > 1:
        ax.hist(d1, bins=25, alpha=0.6, color="#e74c3c", label="Pre-crash (1)", density=True)

    ax.set_title(feat, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle("Feature Distributions: Pre-Crash vs. Normal", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "feature_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR}/feature_distributions.png")

print("\n" + "="*60)
print("FEATURE ENGINEERING COMPLETE")
print("="*60)
print(f"Dataset shape: {dataset.shape}")
print(f"Features: {FEATURE_COLS}")
print("\nNext step: python 03_modeling.py")
