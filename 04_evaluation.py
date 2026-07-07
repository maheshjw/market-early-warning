"""
04_evaluation.py
----------------
Comprehensive evaluation of trained models including:
  1. Precision-Recall curves (primary metric for imbalanced classification)
  2. Threshold analysis (precision-recall tradeoff for business decisions)
  3. Lead-time analysis (how many months of advance warning does the model give?)
  4. SHAP feature importance (interpretability for financial decision-makers)
  5. Portfolio backtest (hedged strategy vs. buy-and-hold)
  6. Confusion matrices at optimal threshold
  7. Historical signal overlay plot

Author: Mahesh Jagdish Wadhokar
BA889 — Boston University Questrom, June 2026

Methodology notes:
  - PR-AUC is the primary metric (better than ROC-AUC for imbalanced classes)
  - Lead time defined as: months between first model signal and actual crash onset
  - Backtest hedges 50% of equity exposure when P(crash) > threshold
  - SHAP values explain individual predictions (Lundberg & Lee, 2017)
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score, confusion_matrix, f1_score
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("WARNING: shap not installed. SHAP plots disabled. pip install shap")
    SHAP_AVAILABLE = False

PROC_DIR  = "data/processed"
MODEL_DIR = "models"
FIG_DIR   = "reports/figures"
RAW_DIR   = "data/raw"

# ── Load data ──────────────────────────────────────────────────────────────────
print("[Loading] Reading processed data and models...")
dataset = pd.read_csv(
    os.path.join(PROC_DIR, "dataset_features_labels.csv"),
    index_col=0, parse_dates=True
)
FEATURE_COLS = [c for c in dataset.columns if c != "crash_label"]
X = dataset[FEATURE_COLS]
y = dataset["crash_label"]

train_mask = dataset.index < "2015-01-01"
test_mask  = dataset.index >= "2015-01-01"
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# Load models
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
fitted_models = {}
for f in model_files:
    name = f.replace(".pkl", "").replace("_", " ").title()
    fitted_models[name] = joblib.load(os.path.join(MODEL_DIR, f))

# Load predicted probabilities
proba_df     = pd.read_csv(os.path.join(PROC_DIR, "test_probabilities.csv"),
                            index_col=0, parse_dates=True)
proba_df_all = pd.read_csv(os.path.join(PROC_DIR, "all_probabilities.csv"),
                            index_col=0, parse_dates=True)

# Load S&P 500 for backtest
sp500_raw = pd.read_csv(os.path.join(RAW_DIR, "sp500.csv"), index_col=0, parse_dates=True)
sp500_m   = sp500_raw.squeeze().resample("ME").last()

print(f"  Models loaded: {list(fitted_models.keys())}")
print(f"  Test period:   {X_test.index.min().date()} → {X_test.index.max().date()}")

COLORS = {
    "Logistic Regression": "#3498db",
    "Random Forest"       : "#2ecc71",
    "Xgboost"             : "#e74c3c",
}
DEFAULT_COLOR = "#9b59b6"


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Precision-Recall Curves
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Precision-Recall curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax_pr  = axes[0]
ax_roc = axes[1]

for name in proba_df.columns:
    y_proba = proba_df[name].values
    color   = COLORS.get(name, DEFAULT_COLOR)

    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    ax_pr.plot(recall, precision, color=color, linewidth=2,
               label=f"{name} (AP={pr_auc:.3f})")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={roc_auc:.3f})")

# Baseline (random classifier)
baseline = y_test.mean()
ax_pr.axhline(baseline, linestyle="--", color="gray", linewidth=1,
              label=f"Baseline (crash rate={baseline:.2f})")
ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")

ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=11)
ax_pr.set_ylabel("Precision (PPV)", fontsize=11)
ax_pr.set_title("Precision-Recall Curve\n(Primary metric — imbalanced classes)", fontsize=11, fontweight="bold")
ax_pr.legend(fontsize=9)
ax_pr.grid(True, alpha=0.3)
ax_pr.set_xlim([0, 1])
ax_pr.set_ylim([0, 1])

ax_roc.set_xlabel("False Positive Rate", fontsize=11)
ax_roc.set_ylabel("True Positive Rate", fontsize=11)
ax_roc.set_title("ROC Curve", fontsize=11, fontweight="bold")
ax_roc.legend(fontsize=9)
ax_roc.grid(True, alpha=0.3)

plt.suptitle("Model Performance on Holdout Set (2015–2024)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pr_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR}/pr_curves.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Lead-Time Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] Lead-time analysis...")

def compute_lead_times(y_true: pd.Series, y_proba: pd.Series,
                        threshold: float = 0.5) -> list:
    """
    For each actual crash period, find when the model FIRST fired a signal
    above threshold. Return the number of months of advance warning.
    
    This is the key novel metric: how early does the model warn?
    """
    y_signal   = (y_proba >= threshold).astype(int)
    lead_times = []

    # Find crash onset dates (transitions from 0→1 in actual labels)
    label_arr  = y_true.values
    crash_onsets = []
    for i in range(1, len(label_arr)):
        if label_arr[i] == 1 and label_arr[i-1] == 0:
            crash_onsets.append(y_true.index[i])

    for onset in crash_onsets:
        # Look back up to 12 months for the first signal
        lookback_start = onset - pd.DateOffset(months=12)
        window = y_signal.loc[lookback_start:onset]
        signal_dates = window[window == 1].index
        if len(signal_dates) > 0:
            first_signal  = signal_dates[0]
            months_before = (onset.year - first_signal.year) * 12 + \
                            (onset.month - first_signal.month)
            lead_times.append(months_before)

    return lead_times

fig, axes = plt.subplots(1, len(proba_df.columns), figsize=(5 * len(proba_df.columns), 5))
if len(proba_df.columns) == 1:
    axes = [axes]

for ax, name in zip(axes, proba_df.columns):
    color      = COLORS.get(name, DEFAULT_COLOR)
    y_proba_s  = proba_df[name]
    lead_times = compute_lead_times(y_test, y_proba_s, threshold=0.5)

    if lead_times:
        ax.hist(lead_times, bins=range(0, 13), color=color, alpha=0.8,
                edgecolor="white", linewidth=0.5)
        ax.axvline(np.mean(lead_times), color="black", linestyle="--", linewidth=1.5,
                   label=f"Mean: {np.mean(lead_times):.1f} mo")
        ax.set_xlabel("Months of Advance Warning", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 12])
        print(f"  {name}: mean lead time = {np.mean(lead_times):.1f} months (n={len(lead_times)} crashes)")
    else:
        ax.text(0.5, 0.5, "No signals detected\nat this threshold",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(name, fontsize=10)

plt.suptitle("Lead-Time Distribution: Months of Advance Warning Before Crash",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "lead_time_hist.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR}/lead_time_hist.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: SHAP Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] SHAP feature importance...")

if SHAP_AVAILABLE:
    # Use best model for SHAP (XGBoost preferred, else Random Forest)
    best_name = "Xgboost" if "Xgboost" in fitted_models else \
                ("Random Forest" if "Random Forest" in fitted_models else
                 list(fitted_models.keys())[0])
    best_model = fitted_models[best_name]

    # Extract the classifier from the pipeline
    clf     = best_model.named_steps["clf"]
    scaler  = best_model.named_steps["scaler"]
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=FEATURE_COLS)

    try:
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_scaled_df)

        # For binary classifiers, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Summary plot (beeswarm)
        plt.sca(axes[0])
        shap.summary_plot(shap_vals, X_test_scaled_df, feature_names=FEATURE_COLS,
                          show=False, plot_type="dot", max_display=12)
        axes[0].set_title(f"SHAP Summary — {best_name}\n(impact on pre-crash probability)",
                          fontsize=10, fontweight="bold")

        # Bar plot (mean absolute SHAP)
        mean_shap = np.abs(shap_vals).mean(axis=0)
        sorted_idx = np.argsort(mean_shap)
        plt.sca(axes[1])
        axes[1].barh(np.array(FEATURE_COLS)[sorted_idx], mean_shap[sorted_idx],
                     color="#e74c3c", alpha=0.8)
        axes[1].set_xlabel("Mean |SHAP value|", fontsize=10)
        axes[1].set_title("Feature Importance\n(Mean Absolute SHAP)", fontsize=10, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {FIG_DIR}/shap_summary.png")

        # Save SHAP values
        shap_df = pd.DataFrame(shap_vals, columns=FEATURE_COLS, index=X_test.index)
        shap_df.to_csv(os.path.join(PROC_DIR, "shap_values.csv"))

    except Exception as e:
        print(f"  SHAP error: {e}")

    # For Logistic Regression — use linear coefficient plot instead
    if "Logistic Regression" in fitted_models:
        lr_model  = fitted_models["Logistic Regression"]
        lr_coef   = lr_model.named_steps["clf"].coef_[0]
        lr_sorted = np.argsort(np.abs(lr_coef))

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#e74c3c" if c > 0 else "#3498db" for c in lr_coef[lr_sorted]]
        ax.barh(np.array(FEATURE_COLS)[lr_sorted], lr_coef[lr_sorted],
                color=colors, alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient (standardized)", fontsize=10)
        ax.set_title("Logistic Regression Coefficients\n(Red=increases crash probability, Blue=decreases)",
                     fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "lr_coefficients.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {FIG_DIR}/lr_coefficients.png")

else:
    # Fallback: Random Forest feature importance
    print("  SHAP not available — using RF feature importance as fallback.")
    if "Random Forest" in fitted_models:
        rf = fitted_models["Random Forest"].named_steps["clf"]
        importances = rf.feature_importances_
        sorted_idx  = np.argsort(importances)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(np.array(FEATURE_COLS)[sorted_idx], importances[sorted_idx],
                color="#2ecc71", alpha=0.8)
        ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=10)
        ax.set_title("Random Forest Feature Importance\n(Install shap for SHAP values)",
                     fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {FIG_DIR}/shap_summary.png (RF importance fallback)")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Portfolio Backtest
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Portfolio backtest...")

def backtest_strategy(sp500_monthly: pd.Series,
                       signal_proba:  pd.Series,
                       threshold:     float = 0.5,
                       hedge_fraction: float = 0.5) -> pd.DataFrame:
    """
    Simple backtest: when model signals P(crash) > threshold,
    reduce equity exposure by hedge_fraction (e.g., 0.5 = 50% in cash/bonds).
    
    Returns monthly returns for: strategy, buy-and-hold, and cash.
    """
    sp500_ret = sp500_monthly.pct_change().fillna(0)

    # Align signal to same index
    signal_aligned = signal_proba.reindex(sp500_ret.index, method="ffill").fillna(0)
    in_market      = 1 - (signal_aligned >= threshold).astype(float) * hedge_fraction

    strategy_ret  = sp500_ret * in_market  # partial exposure
    bh_ret        = sp500_ret              # full exposure
    cash_ret       = pd.Series(0.002, index=sp500_ret.index)  # ~2.4% annual cash return

    # Cumulative returns
    cum_strategy = (1 + strategy_ret).cumprod()
    cum_bh       = (1 + bh_ret).cumprod()

    result = pd.DataFrame({
        "Strategy": cum_strategy,
        "Buy & Hold": cum_bh,
        "Signal (hedge when >0.5)": signal_aligned,
    })
    return result, strategy_ret, bh_ret

# Use best model's probabilities over full period
best_col = "Xgboost" if "Xgboost" in proba_df_all.columns else proba_df_all.columns[0]
proba_full  = proba_df_all[best_col]
sp500_aligned = sp500_m.loc[proba_full.index]

backtest_df, strat_ret, bh_ret = backtest_strategy(sp500_aligned, proba_full)

# Compute Sharpe ratios (annualized, assuming risk-free = 2.4%/yr = 0.2%/mo)
def sharpe(monthly_returns, rf_monthly=0.002):
    excess = monthly_returns - rf_monthly
    return (excess.mean() / excess.std()) * np.sqrt(12)

sharpe_strategy = sharpe(strat_ret)
sharpe_bh       = sharpe(bh_ret)

# Max drawdown
def max_drawdown(cum_returns):
    peak = cum_returns.cummax()
    dd   = (cum_returns - peak) / peak
    return dd.min()

mdd_strategy = max_drawdown(backtest_df["Strategy"])
mdd_bh       = max_drawdown(backtest_df["Buy & Hold"])

print(f"  Sharpe — Strategy: {sharpe_strategy:.2f}  |  Buy & Hold: {sharpe_bh:.2f}")
print(f"  Max DD — Strategy: {mdd_strategy:.1%}  |  Buy & Hold: {mdd_bh:.1%}")

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

ax1 = axes[0]
ax1.plot(backtest_df.index, backtest_df["Buy & Hold"], color="#2980b9", linewidth=1.5, label="Buy & Hold")
ax1.plot(backtest_df.index, backtest_df["Strategy"],  color="#e74c3c", linewidth=1.5, label=f"Model-Hedged Strategy")
ax1.set_ylabel("Cumulative Return (base=1)", fontsize=10)
ax1.set_title(
    f"Portfolio Backtest — {best_col} Model  |  "
    f"Sharpe: Strategy={sharpe_strategy:.2f} vs B&H={sharpe_bh:.2f}  |  "
    f"Max DD: Strategy={mdd_strategy:.1%} vs B&H={mdd_bh:.1%}",
    fontsize=10, fontweight="bold"
)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.fill_between(backtest_df.index, 0, backtest_df["Signal (hedge when >0.5)"],
                 color="#e74c3c", alpha=0.5, label="Crash probability signal")
ax2.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Threshold=0.5")
ax2.set_ylabel("P(crash within 6 months)", fontsize=10)
ax2.set_xlabel("Date", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "backtest.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR}/backtest.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Confusion Matrices
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/6] Confusion matrices...")

fig, axes = plt.subplots(1, len(proba_df.columns), figsize=(5 * len(proba_df.columns), 4))
if len(proba_df.columns) == 1:
    axes = [axes]

for ax, name in zip(axes, proba_df.columns):
    y_proba = proba_df[name].values
    # Find threshold that maximizes F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    opt_threshold = thresholds[np.argmax(f1s[:-1])] if len(thresholds) > 0 else 0.5
    y_pred = (y_proba >= opt_threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred:Normal","Pred:Pre-Crash"],
                yticklabels=["True:Normal","True:Pre-Crash"],
                ax=ax, cbar=False)
    ax.set_title(f"{name}\n(threshold={opt_threshold:.2f}, F1={f1_score(y_test, y_pred):.3f})",
                 fontsize=9, fontweight="bold")

plt.suptitle("Confusion Matrices at Optimal F1 Threshold (2015-2024 Holdout)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR}/confusion_matrix.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Historical signal overlay
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/6] Historical signal overlay...")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

sp500_full = sp500_m.loc["2000-01-01":]

# Shade known crash periods for reference
CRASHES = [
    ("2000-03-01", "2002-10-01", "Dot-com\nbust"),
    ("2007-10-01", "2009-03-01", "GFC"),
    ("2020-02-01", "2020-04-01", "COVID-19"),
    ("2022-01-01", "2022-10-01", "Rate-driven\nselloff"),
]

for ax in axes:
    for start, end, label in CRASHES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.15, color="#e74c3c", zorder=0)
        ax.text(pd.Timestamp(start), ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                label, fontsize=7, color="#c0392b", va="top")

sp500_plot = sp500_full.loc[proba_df_all.index.intersection(sp500_full.index)]
axes[0].plot(sp500_plot.index, sp500_plot.values, color="#1a1a2e", linewidth=1.2, label="S&P 500")
axes[0].set_ylabel("S&P 500 Price")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_title("S&P 500 with Known Crash Periods Highlighted", fontsize=10, fontweight="bold")

for name, color in COLORS.items():
    if name in proba_df_all.columns:
        axes[1].plot(proba_df_all.index, proba_df_all[name],
                     color=color, linewidth=1.2, alpha=0.8, label=name)

axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Threshold=0.5")
axes[1].fill_between(proba_df_all.index, 0.5, 1,
                     where=proba_df_all[best_col] > 0.5,
                     alpha=0.1, color="#e74c3c", label="Active warning zone")
axes[1].set_ylabel("P(crash within 6 months)")
axes[1].set_xlabel("Date")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 1])
axes[1].set_title("Model Crash Probability Over Time", fontsize=10, fontweight="bold")

plt.suptitle("Early Warning System — Historical Signal Reconstruction",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "historical_signal.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR}/historical_signal.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary Report
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)

holdout_df = pd.read_csv(os.path.join(PROC_DIR, "holdout_results.csv"), index_col=0)
cv_df      = pd.read_csv(os.path.join(PROC_DIR, "cv_results.csv"),      index_col=0)

print("\nCross-Validation Results (training set, TimeSeriesSplit k=5):")
print(cv_df[["pr_auc_mean", "pr_auc_std", "roc_auc_mean", "f1_mean"]].round(3).to_string())

print("\nHoldout Results (test set 2015-2024):")
print(holdout_df.round(3).to_string())

print(f"\nBacktest Summary ({best_col} model):")
print(f"  Strategy Sharpe:    {sharpe_strategy:.3f}")
print(f"  Buy&Hold Sharpe:    {sharpe_bh:.3f}")
print(f"  Strategy Max DD:    {mdd_strategy:.1%}")
print(f"  Buy&Hold Max DD:    {mdd_bh:.1%}")

print(f"\nAll figures saved to: {FIG_DIR}/")
print("  eda_overview.png")
print("  feature_correlation.png")
print("  feature_distributions.png")
print("  pr_curves.png")
print("  lead_time_hist.png")
print("  shap_summary.png")
print("  backtest.png")
print("  confusion_matrix.png")
print("  historical_signal.png")

print("\nNext step: streamlit run app/dashboard.py")
