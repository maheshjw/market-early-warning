"""
app/dashboard.py
----------------
Streamlit web dashboard for live macroeconomic crash risk monitoring.

Features:
  - Live risk score (0-1) updated daily via FRED + yfinance APIs
  - Top 3 SHAP drivers explaining today's signal
  - Historical signal overlay against S&P 500
  - Individual feature gauge charts
  - Model performance summary tab

Run locally:  streamlit run app/dashboard.py
Deploy:       Push to GitHub → connect at streamlit.io/cloud
              Set FRED_API_KEY in Streamlit Cloud secrets manager

Author: Mahesh Jagdish Wadhokar
BA889 — Boston University Questrom, June 2026
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings("ignore")

# Resolve project root — works whether run via `streamlit run` or inside Jupyter
try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(_THIS_DIR)
except NameError:
    # __file__ not defined in Jupyter — use current working directory
    PROJECT_ROOT = os.getcwd()
    # If running from app/ subdirectory, go up one level
    if os.path.basename(PROJECT_ROOT) == "app":
        PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

try:
    import streamlit as st
except ImportError:
    raise ImportError("Run: pip install streamlit")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Equity Market Early Warning System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
PROC_DIR  = os.path.join(PROJECT_ROOT, "data", "processed")
FIG_DIR   = os.path.join(PROJECT_ROOT, "reports", "figures")
FEATURE_COLS = [
    "yield_2y10y", "yield_3m10y", "hy_spread", "hy_spread_vel",
    "vix_level", "vix_change_30d", "aaii_bull_bear", "trends_fear",
    "claims_ma4_chg", "cfnai_ma3", "lei_mom", "sector_rotation"
]

FEATURE_DESCRIPTIONS = {
    "yield_2y10y"    : "10yr - 2yr Treasury Yield Spread (%)",
    "yield_3m10y"    : "10yr - 3mo Treasury Yield Spread (%)",
    "hy_spread"      : "High-Yield Credit Spread (%)",
    "hy_spread_vel"  : "HY Spread Velocity (3mo change)",
    "vix_level"      : "VIX — Fear Index",
    "vix_change_30d" : "VIX 30-day % Change",
    "aaii_bull_bear" : "AAII Bull-Bear Sentiment Spread",
    "trends_fear"    : "Google Trends Fear Composite (0-100)",
    "claims_ma4_chg" : "Initial Claims 4-week MA % Change",
    "cfnai_ma3"      : "CFNAI 3-Month Moving Average",
    "lei_mom"        : "Leading Economic Index MoM % Change",
    "sector_rotation": "Defensive vs Cyclical ETF Momentum"
}

# Color mapping for risk levels
def risk_color(score: float) -> str:
    if score < 0.3:  return "#27ae60"   # green
    elif score < 0.5: return "#f39c12"  # yellow
    elif score < 0.7: return "#e67e22"  # orange
    else:             return "#e74c3c"  # red

def risk_label(score: float) -> str:
    if score < 0.3:  return "LOW"
    elif score < 0.5: return "MODERATE"
    elif score < 0.7: return "ELEVATED"
    else:             return "HIGH"


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)  # cache for 1 hour
def load_live_data(fred_api_key: str) -> pd.DataFrame:
    """
    Pull current macro data from FRED + yfinance and engineer features
    for live risk scoring.
    """
    if not FRED_AVAILABLE or not YFINANCE_AVAILABLE:
        return None

    try:
        fred = Fred(api_key=fred_api_key)

        # Pull recent 18 months for feature computation
        end   = pd.Timestamp.today().strftime("%Y-%m-%d")
        start = (pd.Timestamp.today() - pd.DateOffset(months=18)).strftime("%Y-%m-%d")

        live = pd.DataFrame()

        # Yield spreads
        live["yield_2y10y"] = fred.get_series("T10Y2Y", observation_start=start).resample("ME").last()
        live["yield_3m10y"] = fred.get_series("T10Y3M", observation_start=start).resample("ME").last()

        # HY credit spread
        hy_raw = fred.get_series("BAMLH0A0HYM2", observation_start=start).resample("ME").last()
        live["hy_spread"]     = hy_raw
        live["hy_spread_vel"] = hy_raw.diff(3)

        # VIX
        vix_raw = yf.download("^VIX", start=start, end=end, progress=False)["Adj Close"]
        vix_m   = vix_raw.resample("ME").mean()
        live["vix_level"]     = vix_m
        live["vix_change_30d"] = vix_m.pct_change(1) * 100

        # CFNAI + LEI
        live["cfnai_ma3"] = fred.get_series("CFNAI", observation_start=start).resample("ME").last().rolling(3).mean()
        lei_raw           = fred.get_series("USSLIND", observation_start=start).resample("ME").last()
        live["lei_mom"]   = lei_raw.pct_change(1) * 100

        # Initial claims
        claims_raw = fred.get_series("ICSA", observation_start=start)
        claims_ma4 = claims_raw.rolling(4).mean()
        live["claims_ma4_chg"] = claims_ma4.pct_change(1).resample("ME").last() * 100

        # Sector rotation (defensive vs cyclical)
        etfs = {}
        for ticker, label in [("XLU","d1"),("XLP","d2"),("XLY","c1"),("XLK","c2")]:
            s = yf.download(ticker, start=start, end=end, progress=False)["Adj Close"]
            etfs[label] = s.resample("ME").last()

        defensive = ((etfs["d1"] + etfs["d2"]) / 2).pct_change(3)
        cyclical  = ((etfs["c1"] + etfs["c2"]) / 2).pct_change(3)
        live["sector_rotation"] = (defensive - cyclical) * 100

        # AAII placeholder (requires manual update)
        live["aaii_bull_bear"] = np.nan
        live["trends_fear"]    = np.nan

        live = live.dropna(how="all")
        return live

    except Exception as e:
        return None


@st.cache_data
def load_historical() -> pd.DataFrame:
    path = os.path.join(PROC_DIR, "dataset_features_labels.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


@st.cache_resource
def load_models():
    models = {}
    if not os.path.exists(MODEL_DIR):
        return models
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl"):
            name = f.replace(".pkl", "").replace("_", " ").title()
            models[name] = joblib.load(os.path.join(MODEL_DIR, f))
    return models


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/BU_Terrier_Head_logo.svg/120px-BU_Terrier_Head_logo.svg.png",
             width=60)
    st.title("⚠️ Market Early Warning")
    st.caption("BA889 Research Project\nBoston University Questrom")
    st.divider()

    fred_key = st.text_input(
        "FRED API Key",
        type="password",
        placeholder="Enter your FRED API key",
        help="Get a free key at fred.stlouisfed.org/docs/api/api_key.html"
    )
    live_mode = st.toggle("Live Data Mode", value=False,
                          help="Pulls current data from FRED/yfinance. Requires API key.")
    threshold = st.slider("Alert Threshold", 0.1, 0.9, 0.5, 0.05,
                           help="Probability above which model flags elevated risk")
    selected_model = st.selectbox("Model", ["XGBoost", "Random Forest", "Logistic Regression"])
    st.divider()
    st.caption("Data: FRED, yfinance, AAII, Google Trends\nMethod: XGBoost + TimeSeriesSplit CV")


# ── Main content ──────────────────────────────────────────────────────────────
st.title("📉 Equity Market Early Warning System")
st.markdown(
    "Machine learning-based detection of early warning signals for significant S&P 500 downturns "
    "(≥15% peak-to-trough within 6 months) using macroeconomic, sentiment, and alternative data."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🚨 Live Risk Score", "📊 Historical Analysis", "🔍 Feature Explorer", "📋 Model Performance"
])


# ── Tab 1: Live Risk Score ────────────────────────────────────────────────────
with tab1:
    st.subheader("Current Macroeconomic Risk Score")

    models = load_models()
    historical = load_historical()

    if not models:
        st.warning("No trained models found. Run `python 03_modeling.py` first.")
    else:
        # Live data or last historical reading
        if live_mode and fred_key:
            with st.spinner("Fetching live data from FRED & yfinance..."):
                live_df = load_live_data(fred_key)
            if live_df is None:
                st.error("Could not load live data. Check your API key.")
                live_df = historical
                st.info("Falling back to last historical reading.")
        else:
            live_df = historical
            if not live_mode:
                st.info("Toggle 'Live Data Mode' and enter your FRED API key for real-time scoring.")

        if live_df is not None and selected_model in models:
            # Use most recent row
            model = models[selected_model]
            latest_features = live_df[FEATURE_COLS].dropna(how="all").iloc[-1:].ffill().bfill()

            if not latest_features.empty:
                try:
                    risk_score = model.predict_proba(latest_features)[0, 1]
                    as_of_date = latest_features.index[-1].strftime("%B %Y") if hasattr(latest_features.index[-1], 'strftime') else "Latest"

                    col1, col2, col3, col4 = st.columns(4)
                    color = risk_color(risk_score)
                    label = risk_label(risk_score)

                    with col1:
                        st.metric("Risk Score", f"{risk_score:.1%}",
                                  help="Probability of ≥15% drawdown within next 6 months")
                    with col2:
                        st.metric("Risk Level", label)
                    with col3:
                        st.metric("Model", selected_model)
                    with col4:
                        st.metric("As of", as_of_date)

                    # Big risk gauge
                    gauge_html = f"""
                    <div style="text-align:center; padding:20px; background:linear-gradient(135deg,#1a1a2e,#16213e); border-radius:15px; margin:10px 0;">
                        <div style="font-size:72px; font-weight:bold; color:{color};">{risk_score:.1%}</div>
                        <div style="font-size:22px; color:{color}; letter-spacing:4px;">{label} RISK</div>
                        <div style="font-size:13px; color:#aaa; margin-top:8px;">
                            Probability of significant market downturn within 6 months
                        </div>
                        <div style="background:#333; border-radius:10px; height:16px; margin:12px auto; width:80%;">
                            <div style="background:{color}; width:{risk_score*100:.0f}%; height:16px; border-radius:10px; transition:width 0.5s;"></div>
                        </div>
                        <div style="font-size:11px; color:#888;">Alert threshold: {threshold:.0%}  |  Model: {selected_model}  |  As of {as_of_date}</div>
                    </div>
                    """
                    st.html(gauge_html)

                    if risk_score >= threshold:
                        st.error(f"🚨 **ALERT**: Risk score {risk_score:.1%} exceeds threshold {threshold:.0%}. "
                                 "Consider reviewing portfolio positioning.")
                    else:
                        st.success(f"✅ Risk score {risk_score:.1%} is below alert threshold {threshold:.0%}.")

                except Exception as e:
                    st.error(f"Scoring error: {e}")
            else:
                st.warning("No valid feature data available.")


# ── Tab 2: Historical Analysis ────────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns(2)

    hist_signal_path = os.path.join(FIG_DIR, "historical_signal.png")
    if os.path.exists(hist_signal_path):
        with col1:
            st.subheader("Historical Signal vs. S&P 500")
            st.image(hist_signal_path, use_column_width=True)
    else:
        with col1:
            st.info("Run `python 04_evaluation.py` to generate historical signal chart.")

    backtest_path = os.path.join(FIG_DIR, "backtest.png")
    if os.path.exists(backtest_path):
        with col2:
            st.subheader("Portfolio Backtest")
            st.image(backtest_path, use_column_width=True)

    eda_path = os.path.join(FIG_DIR, "eda_overview.png")
    if os.path.exists(eda_path):
        st.subheader("Data Overview — Key Indicators")
        st.image(eda_path, use_column_width=True)


# ── Tab 3: Feature Explorer ───────────────────────────────────────────────────
with tab3:
    st.subheader("Feature Importance & Distributions")

    col1, col2 = st.columns(2)

    shap_path = os.path.join(FIG_DIR, "shap_summary.png")
    if os.path.exists(shap_path):
        with col1:
            st.subheader("SHAP Feature Importance")
            st.image(shap_path, use_column_width=True)
            st.caption("SHAP values show each feature's contribution to the crash probability signal. "
                       "Reference: Lundberg & Lee (2017).")

    dist_path = os.path.join(FIG_DIR, "feature_distributions.png")
    if os.path.exists(dist_path):
        with col2:
            st.subheader("Feature Distributions by Class")
            st.image(dist_path, use_column_width=True)

    corr_path = os.path.join(FIG_DIR, "feature_correlation.png")
    if os.path.exists(corr_path):
        st.subheader("Feature Correlation Matrix")
        st.image(corr_path, use_column_width=True)

    # Feature descriptions table
    st.subheader("Feature Dictionary")
    feat_dict = pd.DataFrame([
        {"Feature": k, "Description": v, "Source": src}
        for (k, v), src in zip(
            FEATURE_DESCRIPTIONS.items(),
            ["FRED","FRED","FRED","FRED","yfinance","yfinance","AAII","Google Trends",
             "FRED","FRED","FRED","yfinance"]
        )
    ])
    st.dataframe(feat_dict, use_container_width=True, hide_index=True)


# ── Tab 4: Model Performance ──────────────────────────────────────────────────
with tab4:
    st.subheader("Model Evaluation — Holdout Set (2015–2024)")

    col1, col2 = st.columns(2)

    pr_path = os.path.join(FIG_DIR, "pr_curves.png")
    if os.path.exists(pr_path):
        with col1:
            st.image(pr_path, use_column_width=True)

    cm_path = os.path.join(FIG_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        with col2:
            st.image(cm_path, use_column_width=True)

    lead_path = os.path.join(FIG_DIR, "lead_time_hist.png")
    if os.path.exists(lead_path):
        st.subheader("Lead-Time Distribution")
        st.image(lead_path, use_column_width=True)
        st.caption("Lead time = months of advance warning before crash onset. "
                   "Values >0 indicate the model fires before the drawdown begins.")

    # Numeric results tables
    cv_path = os.path.join(PROC_DIR, "cv_results.csv")
    ho_path = os.path.join(PROC_DIR, "holdout_results.csv")

    if os.path.exists(cv_path) and os.path.exists(ho_path):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cross-Validation (TimeSeriesSplit, k=5)")
            cv_df = pd.read_csv(cv_path, index_col=0)
            st.dataframe(cv_df.round(3), use_container_width=True)
        with col2:
            st.subheader("Holdout Set Performance")
            ho_df = pd.read_csv(ho_path, index_col=0)
            st.dataframe(ho_df.round(3), use_container_width=True)

    st.subheader("Methodology Notes")
    st.markdown("""
    | Design Choice | Rationale |
    |---|---|
    | **TimeSeriesSplit CV** | Preserves temporal ordering — prevents lookahead bias |
    | **PR-AUC as primary metric** | Better than ROC-AUC when classes are imbalanced (~15-20% crash rate) |
    | **class_weight='balanced'** | Prevents model from ignoring rare crash class |
    | **Forward-looking labels** | Labels the PRE-crash state, not the crash itself — enables lead-time analysis |
    | **Pre-2015 train / 2015+ test** | Out-of-sample test covers 2 major regimes (2020 COVID, 2022 rate hike) |
    | **SHAP values** | Lundberg & Lee (2017) — model-agnostic feature attribution |
    """)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "BA889 Research Paper | Mahesh Jagdish Wadhokar | Boston University Questrom School of Business | June 2026  \n"
    "Data: FRED (Federal Reserve), yfinance, AAII Sentiment Survey, Google Trends  \n"
    "Academic use only. Not financial advice."
)
