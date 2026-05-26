import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

from constants import SECTOR_COLORS, STOCKS
from data_utils import load_data, load_index_data
from indicators import add_indicators, get_52w, get_signal
from market import is_market_open
from model_utils import (
    calculate_backtest_metrics,
    forecast_confidence_series,
    get_model_artifact_info,
    get_trading_days,
    load_model_and_scaler,
    next_trading_day,
    predict_7days,
    predict_tomorrow,
    run_baseline_backtest,
    run_backtest,
)

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense India",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)
def clean_html(html_str):
    return "\n".join(line.lstrip() for line in html_str.split("\n"))

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #090E1A !important;
    color: #E0E6F0 !important;
}
[data-testid="stSidebar"] {
    background-color: #0D1525 !important;
    border-right: 1px solid #1E2D4A;
}
[data-testid="stHeader"] { background-color: #090E1A !important; }
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }
p, div, span, label { font-family: 'Space Mono', monospace !important; font-size: 0.82rem; }

.metric-card {
    background: linear-gradient(135deg, #0D1525 0%, #111C35 100%);
    border: 1px solid #1E2D4A;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(0,245,160,0.3);
    box-shadow: 0 0 20px rgba(0,245,160,0.05);
    transform: translateY(-2px);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00F5A0, #00D9F5);
}
.metric-card-bull::before { background: linear-gradient(90deg, #00F5A0, #00D9F5); }
.metric-card-bear::before { background: linear-gradient(90deg, #FF4D6D, #FF8C42); }
.metric-card-neutral::before { background: linear-gradient(90deg, #F5C518, #FF8C42); }

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #5A7499;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #E0E6F0;
    line-height: 1;
}
.metric-sub { font-family: 'Space Mono', monospace; font-size: 0.72rem; margin-top: 6px; }
.bullish { color: #00F5A0; }
.bearish { color: #FF4D6D; }
.neutral { color: #F5C518; }

.signal-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 50px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-buy  { background: rgba(0,245,160,0.12); border: 1px solid #00F5A0; color: #00F5A0; }
.badge-sell { background: rgba(255,77,109,0.12); border: 1px solid #FF4D6D; color: #FF4D6D; }
.badge-hold { background: rgba(245,197,24,0.12); border: 1px solid #F5C518; color: #F5C518; }

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    color: #5A7499;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    border-bottom: 1px solid #1E2D4A;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

[data-testid="stSelectbox"] > div > div {
    background-color: #0D1525 !important;
    border: 1px solid #1E2D4A !important;
    color: #E0E6F0 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
}
.stSelectbox label {
    color: #5A7499 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

div.stButton > button {
    background: linear-gradient(135deg, #0D1525, #111C35);
    border: 1px solid #1E2D4A;
    color: #00F5A0;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    border-radius: 8px;
    padding: 8px 16px;
    cursor: pointer;
    width: 100%;
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    border-color: #00F5A0;
    background: rgba(0,245,160,0.08);
    box-shadow: 0 0 12px rgba(0,245,160,0.15);
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: #0D1525 !important;
    border-bottom: 1px solid #1E2D4A !important;
    gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background-color: transparent !important;
    color: #5A7499 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-radius: 6px 6px 0 0 !important;
    padding: 8px 20px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background-color: #111C35 !important;
    color: #00F5A0 !important;
    border-bottom: 2px solid #00F5A0 !important;
}
[data-testid="stTabContent"] {
    background-color: #090E1A !important;
    padding-top: 16px !important;
}

hr { border-color: #1E2D4A !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #090E1A; }
::-webkit-scrollbar-thumb { background: #1E2D4A; border-radius: 2px; }

.model-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    background: rgba(0,245,160,0.08);
    border: 1px solid rgba(0,245,160,0.3);
    color: #00F5A0;
    letter-spacing: 0.08em;
    margin-left: 8px;
    vertical-align: middle;
}
.market-open-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    background: rgba(0,245,160,0.08);
    border: 1px solid rgba(0,245,160,0.3);
    color: #00F5A0;
    letter-spacing: 0.08em;
    margin-left: 8px;
    vertical-align: middle;
    animation: pulse 2s infinite;
}
.market-closed-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    background: rgba(255,77,109,0.08);
    border: 1px solid rgba(255,77,109,0.3);
    color: #FF4D6D;
    letter-spacing: 0.08em;
    margin-left: 8px;
    vertical-align: middle;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.warn-box {
    background: rgba(245,197,24,0.05);
    border: 1px solid rgba(245,197,24,0.2);
    border-radius: 8px;
    padding: 10px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #F5C518;
    margin-top: 16px;
}

.logo-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00F5A0, #00D9F5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 4px;
}
.logo-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #3A5070;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.forecast-card {
    background: linear-gradient(135deg, #0D1525 0%, #111C35 100%);
    border: 1px solid #1E2D4A;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.2s ease;
}
.forecast-card:hover {
    border-color: rgba(0,245,160,0.3);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.forecast-day {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #5A7499;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.forecast-price {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #E0E6F0;
}
.forecast-change { font-family: 'Space Mono', monospace; font-size: 0.65rem; margin-top: 4px; }

.stat-box {
    background: #0D1525;
    border: 1px solid #1E2D4A;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    transition: all 0.2s ease;
}
.stat-box:hover { border-color: rgba(0,245,160,0.2); }
.stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #5A7499;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #E0E6F0;
    margin-top: 2px;
}

.about-card {
    background: linear-gradient(135deg, #0D1525 0%, #111C35 100%);
    border: 1px solid #1E2D4A;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.about-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: #00F5A0;
    margin-bottom: 8px;
}
.about-text {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #5A7499;
    line-height: 1.8;
}

.index-card {
    background: #0D1525;
    border: 1px solid #1E2D4A;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
    margin-bottom: 8px;
}
.index-name { font-family: 'Space Mono', monospace; font-size: 0.6rem; color: #5A7499; text-transform: uppercase; letter-spacing: 0.1em; }
.index-value { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; color: #E0E6F0; margin: 2px 0; }
.index-change { font-family: 'Space Mono', monospace; font-size: 0.62rem; }

.sidebar-stock-info {
    background: rgba(0,245,160,0.05);
    border: 1px solid rgba(0,245,160,0.15);
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #5A7499;
}

.accuracy-bar-wrap {
    background: #1E2D4A;
    border-radius: 4px;
    height: 6px;
    margin-top: 6px;
    overflow: hidden;
}
.accuracy-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #00F5A0, #00D9F5);
}

.footer {
    text-align: center;
    padding: 24px 0 12px 0;
    border-top: 1px solid #1E2D4A;
    margin-top: 32px;
}
.footer-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00F5A0, #00D9F5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.footer-text { font-family: 'Space Mono', monospace; font-size: 0.62rem; color: #1E2D4A; margin-top: 6px; }
.footer-links { font-family: 'Space Mono', monospace; font-size: 0.62rem; color: #3A5070; margin-top: 8px; }
.footer-links a { color: #3A5070; text-decoration: none; margin: 0 8px; }
.footer-links a:hover { color: #00F5A0; }
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY THEME ────────────────────────────────────────────────────────────────
PLOT_BG  = "#090E1A"
GRID_CLR = "#1E2D4A"
TEXT_CLR = "#5A7499"
GREEN    = "#00F5A0"
RED      = "#FF4D6D"
BLUE     = "#00D9F5"
YELLOW   = "#F5C518"
PURPLE   = "#B06DFF"

CHART_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["autoScale2d", "lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "filename": "StockSense_India"},
}

def base_layout(title=""):
    return dict(
        title=dict(text=title, font=dict(family="Syne", size=13, color="#E0E6F0")),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Space Mono", color=TEXT_CLR, size=10),
        xaxis=dict(
            gridcolor=GRID_CLR, showgrid=True, zeroline=False,
            tickfont=dict(size=9), rangeslider=dict(visible=False),
            showspikes=True, spikemode="across", spikedash="dot",
            spikecolor="#3A5070", spikethickness=1,
            rangeselector=dict(
                buttons=[
                    dict(count=1,  label="1M", step="month", stepmode="backward"),
                    dict(count=3,  label="3M", step="month", stepmode="backward"),
                    dict(count=6,  label="6M", step="month", stepmode="backward"),
                    dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                    dict(count=2,  label="2Y", step="year",  stepmode="backward"),
                    dict(step="all", label="MAX")
                ],
                bgcolor="#0D1525", activecolor="#00F5A0",
                font=dict(color="#E0E6F0", size=9, family="Space Mono"),
                bordercolor="#1E2D4A", borderwidth=1, x=0, y=1.04
            )
        ),
        yaxis=dict(
            gridcolor=GRID_CLR, showgrid=True, zeroline=False,
            tickfont=dict(size=9),
            showspikes=True, spikemode="across", spikedash="dot",
            spikecolor="#3A5070", spikethickness=1,
        ),
        margin=dict(l=12, r=12, t=50, b=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0D1525",
            bordercolor="#1E2D4A",
            font=dict(family="Space Mono", size=10, color="#E0E6F0")
        ),
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color="#3A5070",
            activecolor="#00F5A0",
        ),
    )

# ── TOP INDEX BAR ────────────────────────────────────────────
indices = load_index_data()

index_html = '<div style="display:flex;gap:12px;padding:10px 0;overflow-x:auto;border-bottom:1px solid #1E2D4A;margin-bottom:16px;">'
for idx_name, idx_data in indices.items():
    if idx_data["price"] == 0:
        continue
    chg_color = "#00F5A0" if idx_data["change"] >= 0 else "#FF4D6D"
    chg_sym   = "▲" if idx_data["change"] >= 0 else "▼"
    index_html += f"""
    <div style="background:#0D1525;border:1px solid #1E2D4A;border-radius:8px;
                padding:8px 16px;white-space:nowrap;min-width:140px;">
        <div style="font-family:Space Mono;font-size:0.6rem;color:#5A7499;
                    text-transform:uppercase;letter-spacing:0.1em;">{idx_name}</div>
        <div style="font-family:Syne,sans-serif;font-size:0.95rem;
                    font-weight:700;color:#E0E6F0;">{idx_data['price']:,.2f}</div>
        <div style="font-family:Space Mono;font-size:0.62rem;color:{chg_color};">
            {chg_sym} {abs(idx_data['pct']):.2f}%</div>
    </div>"""

# Add market status + logo in same bar
market_open = is_market_open()
mbadge_color = "#00F5A0" if market_open else "#FF4D6D"
mbadge_text  = "● Market Open" if market_open else "● Market Closed"
index_html += f"""
    <div style="margin-left:auto;display:flex;align-items:center;gap:12px;">
        <span style="font-family:Space Mono;font-size:0.68rem;
                     color:{mbadge_color};">{mbadge_text}</span>
        <div style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;
                    background:linear-gradient(90deg,#00F5A0,#00D9F5);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            StockSense</div>
    </div>"""
index_html += '</div>'
st.markdown(index_html, unsafe_allow_html=True)


# ── LOAD DATA & MODEL ────────────────────────────────────────
col_range, col_refresh = st.columns([4, 1])

with col_range:
    range_option = st.selectbox(
        "Chart Range",
        ["3 Months", "6 Months", "1 Year", "2 Years", "Max"],
        index=2,
        key="range_selector"
    )
    range_map = {"3 Months": 90, "6 Months": 180, "1 Year": 365, "2 Years": 730, "Max": 9999}
    days_back = range_map[range_option]

with col_refresh:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↻ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── STOCK SELECTION STATE ────────────────────────────────────
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "Reliance Industries"

# Ensure selected stock is in STOCKS keys
if st.session_state["selected_stock"] not in STOCKS:
    st.session_state["selected_stock"] = list(STOCKS.keys())[0]

stock_name = st.session_state["selected_stock"]

@st.cache_data(ttl=300)
def get_stock_summary(ticker):
    import yfinance as yf
    try:
        data = yf.download(ticker, period="5d", progress=False, timeout=10)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        curr_p = float(data["Close"].iloc[-1])
        prev_p = float(data["Close"].iloc[-2]) if len(data) >= 2 else curr_p
        chg_p  = curr_p - prev_p
        pct_p  = (chg_p / prev_p) * 100 if prev_p != 0 else 0.0
        return {"current": curr_p, "change": chg_p, "pct": pct_p}
    except Exception:
        return None

# ── MAIN LAYOUT: LEFT STOCK LIST + RIGHT CONTENT ─────────────
left_col, main_col = st.columns([1, 4])

with left_col:
    st.markdown("""
    <div style="font-family:Space Mono;font-size:0.62rem;color:#5A7499;
                text-transform:uppercase;letter-spacing:0.1em;
                border-bottom:1px solid #1E2D4A;padding-bottom:8px;
                margin-bottom:12px;">Stocks</div>
    """, unsafe_allow_html=True)

    for sname, sdata in STOCKS.items():
        summary = get_stock_summary(sdata["ticker"])
        if summary is not None:
            curr_p = summary["current"]
            pct_p = summary["pct"]
            chg_p = summary["change"]
            clr = "#00F5A0" if chg_p >= 0 else "#FF4D6D"
            sym = "▲" if chg_p >= 0 else "▼"
            
            # Predict
            pred_p = None
            try:
                m, s = load_model_and_scaler(sdata["model"])
                if m is not None:
                    full_d = load_data(sdata["ticker"])
                    if full_d is not None and not full_d.empty:
                        pred_p = predict_tomorrow(m, s, full_d)
            except Exception:
                pass

            pred_html = ""
            if pred_p:
                pred_chg = pred_p - curr_p
                pred_clr = "#00F5A0" if pred_chg >= 0 else "#FF4D6D"
                pred_sym = "▲" if pred_chg >= 0 else "▼"
                pred_html = f"""
                <div style="font-family:Space Mono;font-size:0.58rem;
                            color:{pred_clr};margin-top:2px;">
                    🔮 {pred_sym} ₹{pred_p:,.0f}</div>"""

            is_selected = "border-color:#00F5A0;box-shadow:0 0 10px rgba(0,245,160,0.15);" if sname == stock_name else ""
            sc = SECTOR_COLORS.get(sdata["sector"], "#5A7499")
            btn_label = "Analyzing" if sname == stock_name else "View"

            card_html = f"""
            <div style="background:#0D1525;border:1px solid #1E2D4A;{is_selected}
                        border-radius:8px;padding:12px;margin-bottom:6px;">
                <div style="font-family:Syne,sans-serif;font-size:0.8rem;font-weight:700;color:#E0E6F0;margin-bottom:6px;">
                    {sname}
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-family:Space Mono;font-size:0.8rem;font-weight:700;color:#E0E6F0;">₹{curr_p:,.2f}</span>
                    <span style="font-family:Space Mono;font-size:0.65rem;color:{clr};">{sym} {abs(pct_p):.2f}%</span>
                </div>
                {pred_html}
            </div>"""
            
            st.markdown(clean_html(card_html), unsafe_allow_html=True)
            if st.button(btn_label, key=f"btn_{sname}", use_container_width=True):
                st.session_state["selected_stock"] = sname
                st.rerun()
            st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

with main_col:
    # ── STOCK DETAILS & METRICS ──
    ticker     = STOCKS[stock_name]["ticker"]
    model_name = STOCKS[stock_name]["model"]
    sector     = STOCKS[stock_name]["sector"]

    sc     = SECTOR_COLORS.get(sector, "#5A7499")
    mbadge = '<span class="market-open-badge">● LIVE</span>' if market_open else ''
    st.markdown(
        f"## {stock_name} &nbsp;"
        f"<span style='font-size:0.8rem;color:#3A5070;font-family:Space Mono'>{ticker}</span>"
        f"<span class='model-badge'>LSTM</span>"
        f"<span style='font-size:0.7rem;color:{sc};font-family:Space Mono;margin-left:8px;'>● {sector}</span>"
        f"{mbadge}",
        unsafe_allow_html=True
    )

    with st.spinner(f"Fetching {stock_name} data..."):
        df = load_data(ticker)

    if df is None or df.empty:
        st.error("Could not load data. Please try again.")
        st.stop()

    df = add_indicators(df)

    cutoff = pd.Timestamp(datetime.today() - timedelta(days=days_back)) if days_back < 9999 else df["Date"].iloc[0]
    if days_back < 9999:
        df_view = df[df["Date"] >= cutoff].copy()
    else:
        df_view = df.copy()

    latest_close = float(df["Close"].iloc[-1])
    prev_close   = float(df["Close"].iloc[-2])
    day_change   = latest_close - prev_close
    day_pct      = (day_change / prev_close) * 100
    chg_class    = "bullish" if day_change >= 0 else "bearish"
    chg_symbol   = "▲" if day_change >= 0 else "▼"
    w52_low, w52_high = get_52w(df)

    with st.spinner("Loading LSTM model..."):
        model, scaler = load_model_and_scaler(model_name)
    artifact_info = get_model_artifact_info(model_name)

    pred_price = pred_change = pred_pct = pred_day = None
    if model is not None:
        pred_price  = predict_tomorrow(model, scaler, df)
        pred_change = pred_price - latest_close
        pred_pct    = (pred_change / latest_close) * 100
        pred_day    = next_trading_day(datetime.today())

    signal, badge_class, signal_class = get_signal(df)

    latest_data_date = pd.to_datetime(df["Date"].iloc[-1]).strftime("%d %b %Y")
    model_updated = artifact_info["latest_modified_at"]
    model_updated_label = model_updated.strftime("%d %b %Y") if model_updated else "Unavailable"
    artifact_age = artifact_info["oldest_age_days"]
    missing_artifacts = not artifact_info["model"]["exists"] or not artifact_info["scaler"]["exists"]
    if missing_artifacts:
        freshness_color = "#FF4D6D"
        freshness_note = "Missing artifact"
    elif artifact_age is not None and artifact_age <= 30:
        freshness_color = "#00F5A0"
        freshness_note = "Fresh"
    else:
        freshness_color = "#F5C518"
        freshness_note = "Review age"

    st.markdown(f"""
    <div class="warn-box" style="display:flex;flex-wrap:wrap;gap:18px;align-items:center;border-color:rgba(0,217,245,0.18);background:rgba(0,217,245,0.03);">
        <span><b style="color:#E0E6F0">Latest data:</b> {latest_data_date}</span>
        <span><b style="color:#E0E6F0">Model updated:</b> {model_updated_label}</span>
        <span><b style="color:{freshness_color}">{freshness_note}</b></span>
    </div>
    """, unsafe_allow_html=True)

    # ─── KPI ROW ─────────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        cc = "metric-card metric-card-bull" if day_change >= 0 else "metric-card metric-card-bear"
        st.markdown(clean_html(f"""
        <div class="{cc}">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">₹{latest_close:,.2f}</div>
            <div class="metric-sub {chg_class}">{chg_symbol} ₹{abs(day_change):.2f} ({abs(day_pct):.2f}%) today</div>
        </div>"""), unsafe_allow_html=True)

    with c2:
        if pred_price:
            pc = "bullish" if pred_change >= 0 else "bearish"
            pk = "metric-card metric-card-bull" if pred_change >= 0 else "metric-card metric-card-bear"
            ps = "▲" if pred_change >= 0 else "▼"
            st.markdown(clean_html(f"""
            <div class="{pk}">
                <div class="metric-label">Tomorrow's Prediction</div>
                <div class="metric-value">₹{pred_price:,.2f}</div>
                <div class="metric-sub {pc}">{ps} ₹{abs(pred_change):.2f} ({abs(pred_pct):.2f}%) · {pred_day.strftime("%d %b")}</div>
            </div>"""), unsafe_allow_html=True)

    with c3:
        rsi_val   = float(df["RSI"].iloc[-1])
        rsi_class = "bullish" if rsi_val < 40 else ("bearish" if rsi_val > 65 else "neutral")
        rsi_lbl   = "Oversold" if rsi_val < 30 else ("Overbought" if rsi_val > 70 else "Neutral")
        rsi_card  = "metric-card metric-card-bull" if rsi_val < 40 else ("metric-card metric-card-bear" if rsi_val > 65 else "metric-card metric-card-neutral")
        st.markdown(clean_html(f"""
        <div class="{rsi_card}">
            <div class="metric-label">RSI (14)</div>
            <div class="metric-value">{rsi_val:.1f}</div>
            <div class="metric-sub {rsi_class}">{rsi_lbl}</div>
        </div>"""), unsafe_allow_html=True)

    with c4:
        if w52_low and w52_high:
            pos_pct = ((latest_close - w52_low) / (w52_high - w52_low) * 100) if w52_high != w52_low else 50
            st.markdown(clean_html(f"""
            <div class="metric-card">
                <div class="metric-label">52-Week Range</div>
                <div class="metric-value" style="font-size:1rem;">₹{latest_close:,.0f}</div>
                <div style="font-family:Space Mono;font-size:0.62rem;color:#5A7499;margin-top:6px;">
                    ₹{w52_low:,.0f} &nbsp;—&nbsp; ₹{w52_high:,.0f}
                </div>
                <div class="accuracy-bar-wrap">
                    <div class="accuracy-bar-fill" style="width:{pos_pct:.0f}%"></div>
                </div>
            </div>"""), unsafe_allow_html=True)

    with c5:
        st.markdown(clean_html(f"""
        <div class="metric-card">
            <div class="metric-label">Signal</div>
            <div class="metric-value" style="font-size:1.1rem;padding-top:6px;">
                <span class="signal-badge {badge_class}">{signal}</span>
            </div>
            <div class="metric-sub" style="margin-top:10px;color:#3A5070;">RSI + MA cross</div>
        </div>"""), unsafe_allow_html=True)

    # ─── TABS ────────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈  Live Chart", "🔮  7-Day Forecast", "🧪  Backtest", "ℹ️  About"
    ])

    # ════════════ TAB 1 — LIVE CHART ════════════
    with tab1:
        st.markdown('<div class="section-header">Price Chart · Bollinger Bands · Moving Averages</div>',
                    unsafe_allow_html=True)

        fig_price = go.Figure()

        # Initial view range
        start_view = max(cutoff, df["Date"].iloc[0])
        end_view = df["Date"].iloc[-1] + pd.Timedelta(days=5)

        # Candlestick — improved styling
        fig_price.add_trace(go.Candlestick(
            x=df["Date"],
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            increasing=dict(line=dict(color="#00F5A0", width=1.2), fillcolor="rgba(0,245,160,0.75)"),
            decreasing=dict(line=dict(color="#FF4D6D", width=1.2), fillcolor="rgba(255,77,109,0.75)"),
            whiskerwidth=0.3,
            name="OHLC",
        ))

        # Bollinger Bands
        fig_price.add_trace(go.Scatter(
            x=df["Date"], y=df["BB_Upper"],
            line=dict(color="rgba(0,217,245,0.3)", width=1, dash="dot"),
            name="BB Upper",
            hovertemplate="BB Upper: ₹%{y:,.2f}<extra></extra>"
        ))
        fig_price.add_trace(go.Scatter(
            x=df["Date"], y=df["BB_Lower"],
            fill="tonexty", fillcolor="rgba(0,217,245,0.04)",
            line=dict(color="rgba(0,217,245,0.3)", width=1, dash="dot"),
            name="BB Lower",
            hovertemplate="BB Lower: ₹%{y:,.2f}<extra></extra>"
        ))
        fig_price.add_trace(go.Scatter(
            x=df["Date"], y=df["BB_Mid"],
            line=dict(color="rgba(0,217,245,0.5)", width=1),
            name="BB Mid",
            hovertemplate="BB Mid: ₹%{y:,.2f}<extra></extra>"
        ))

        # Moving Averages
        fig_price.add_trace(go.Scatter(
            x=df["Date"], y=df["MA50"],
            line=dict(color=YELLOW, width=1.5),
            name="MA 50",
            hovertemplate="MA50: ₹%{y:,.2f}<extra></extra>"
        ))
        fig_price.add_trace(go.Scatter(
            x=df["Date"], y=df["MA200"],
            line=dict(color="#FF6B9D", width=1.5),
            name="MA 200",
            hovertemplate="MA200: ₹%{y:,.2f}<extra></extra>"
        ))

        # 52W lines
        if w52_high:
            fig_price.add_hline(y=w52_high, line=dict(color="rgba(0,245,160,0.3)", width=1, dash="dot"),
                annotation_text="52W High", annotation_font=dict(color="#00F5A0", size=9, family="Space Mono"))
        if w52_low:
            fig_price.add_hline(y=w52_low, line=dict(color="rgba(255,77,109,0.3)", width=1, dash="dot"),
                annotation_text="52W Low", annotation_font=dict(color="#FF4D6D", size=9, family="Space Mono"))

        # LSTM prediction star
        if pred_price:
            fig_price.add_trace(go.Scatter(
                x=[pred_day], y=[pred_price], mode="markers+text",
                marker=dict(color=GREEN, size=12, symbol="star", line=dict(color="white", width=1.5)),
                text=[f"  ₹{pred_price:,.0f}"],
                textfont=dict(color=GREEN, size=10, family="Space Mono"),
                textposition="middle right", name="LSTM Prediction",
                hovertemplate="LSTM Pred: ₹%{y:,.2f}<extra></extra>"
            ))

        # Get visible Y range for initial scaling (including BB bands and predictions if any)
        visible_min = float(df_view["Low"].min())
        visible_max = float(df_view["High"].max())
        if "BB_Lower" in df_view.columns and not df_view["BB_Lower"].dropna().empty:
            visible_min = min(visible_min, float(df_view["BB_Lower"].min()))
        if "BB_Upper" in df_view.columns and not df_view["BB_Upper"].dropna().empty:
            visible_max = max(visible_max, float(df_view["BB_Upper"].max()))
        if pred_price:
            visible_min = min(visible_min, pred_price)
            visible_max = max(visible_max, pred_price)

        padding = (visible_max - visible_min) * 0.05
        if padding == 0:
            padding = visible_min * 0.05 if visible_min > 0 else 1.0
        y_min = max(0.0, visible_min - padding)
        y_max = visible_max + padding

        l = base_layout()
        l["height"] = 520
        l["xaxis"]["range"] = [start_view, end_view]
        l["xaxis"]["rangeslider"] = dict(visible=False)
        l["dragmode"] = "pan"        # enables pan by default
        l["xaxis"]["fixedrange"] = False
        l["yaxis"]["fixedrange"] = True
        l["yaxis"]["range"] = [y_min, y_max]
        fig_price.update_layout(**l)

        # ── CHART + INDICATOR PANEL SIDE BY SIDE ──
        chart_col, info_col = st.columns([4, 1])

        with chart_col:
            st.plotly_chart(fig_price, use_container_width=True,
                            config={
                                "scrollZoom": True,        # mouse wheel zoom
                                "displayModeBar": True,
                                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                "modeBarButtonsToAdd":    ["drawline", "eraseshape"],
                                "responsive": True
                            })

        with info_col:
            # Live indicator values shown on the right
            st.markdown(clean_html(f"""
            <div style="background:#0D1525;border:1px solid #1E2D4A;border-radius:8px;
                        padding:12px;font-family:Space Mono;font-size:0.62rem;
                        position:sticky;top:0;">
                <div style="color:#5A7499;text-transform:uppercase;letter-spacing:0.1em;
                            border-bottom:1px solid #1E2D4A;padding-bottom:6px;margin-bottom:10px;">
                    Indicators</div>

                <div style="margin-bottom:10px;">
                    <div style="color:#5A7499;font-size:0.58rem;">MA 50</div>
                    <div style="color:#F5C518;font-weight:700;">₹{float(df['MA50'].iloc[-1]):,.2f}</div>
                </div>
                <div style="margin-bottom:10px;">
                    <div style="color:#5A7499;font-size:0.58rem;">MA 200</div>
                    <div style="color:#FF6B9D;font-weight:700;">₹{float(df['MA200'].iloc[-1]):,.2f}</div>
                </div>
                <div style="margin-bottom:10px;">
                    <div style="color:#5A7499;font-size:0.58rem;">BB Upper</div>
                    <div style="color:rgba(0,217,245,0.8);font-weight:700;">₹{float(df['BB_Upper'].iloc[-1]):,.2f}</div>
                </div>
                <div style="margin-bottom:10px;">
                    <div style="color:#5A7499;font-size:0.58rem;">BB Mid</div>
                    <div style="color:rgba(0,217,245,0.6);font-weight:700;">₹{float(df['BB_Mid'].iloc[-1]):,.2f}</div>
                </div>
                <div style="margin-bottom:10px;">
                    <div style="color:#5A7499;font-size:0.58rem;">BB Lower</div>
                    <div style="color:rgba(0,217,245,0.8);font-weight:700;">₹{float(df['BB_Lower'].iloc[-1]):,.2f}</div>
                </div>
                <div style="border-top:1px solid #1E2D4A;padding-top:10px;margin-top:4px;">
                    <div style="color:#5A7499;font-size:0.58rem;">RSI (14)</div>
                    <div style="color:{'#00F5A0' if float(df['RSI'].iloc[-1]) < 40 else '#FF4D6D' if float(df['RSI'].iloc[-1]) > 65 else '#F5C518'};font-weight:700;">
                        {float(df['RSI'].iloc[-1]):.1f}</div>
                </div>
                <div style="margin-top:10px;">
                    <div style="color:#5A7499;font-size:0.58rem;">MACD</div>
                    <div style="color:{'#00F5A0' if float(df['MACD'].iloc[-1]) > float(df['Signal'].iloc[-1]) else '#FF4D6D'};font-weight:700;">
                        {float(df['MACD'].iloc[-1]):,.2f}</div>
                </div>
                <div style="margin-top:10px;">
                    <div style="color:#5A7499;font-size:0.58rem;">Signal</div>
                    <div style="color:#F5C518;font-weight:700;">
                        {float(df['Signal'].iloc[-1]):,.2f}</div>
                </div>
                <div style="margin-top:10px;">
                    <div style="color:#5A7499;font-size:0.58rem;">Open</div>
                    <div style="color:#E0E6F0;font-weight:700;">₹{float(df['Open'].iloc[-1]):,.2f}</div>
                </div>
                <div style="margin-top:10px;">
                    <div style="color:#5A7499;font-size:0.58rem;">High</div>
                    <div style="color:#00F5A0;font-weight:700;">₹{float(df['High'].iloc[-1]):,.2f}</div>
                </div>
                <div style="margin-top:10px;">
                    <div style="color:#5A7499;font-size:0.58rem;">Low</div>
                    <div style="color:#FF4D6D;font-weight:700;">₹{float(df['Low'].iloc[-1]):,.2f}</div>
                </div>
                <div style="margin-top:10px;">
                    <div style="color:#5A7499;font-size:0.58rem;">Volume</div>
                    <div style="color:#E0E6F0;font-weight:700;">{float(df['Volume'].iloc[-1])/1e6:.2f}M</div>
                </div>
            </div>"""), unsafe_allow_html=True)

        st.markdown('<div class="section-header">Indicators</div>', unsafe_allow_html=True)
        col_rsi, col_macd = st.columns(2)

        with col_rsi:
            fig_rsi = go.Figure()

            # Band fill between 30 and 70
            fig_rsi.add_trace(go.Scatter(
                x=df["Date"], y=[70]*len(df),
                line=dict(color="rgba(255,77,109,0.4)", width=1, dash="dot"),
                showlegend=False, hoverinfo="skip"
            ))
            fig_rsi.add_trace(go.Scatter(
                x=df["Date"], y=[30]*len(df),
                fill="tonexty", fillcolor="rgba(0,217,245,0.04)",
                line=dict(color="rgba(0,245,160,0.4)", width=1, dash="dot"),
                showlegend=False, hoverinfo="skip"
            ))

            # RSI line
            fig_rsi.add_trace(go.Scatter(
                x=df["Date"], y=df["RSI"],
                line=dict(color=BLUE, width=1.8),
                name="RSI(14)",
                hovertemplate="RSI: %{y:.1f}<extra></extra>"
            ))

            fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(255,77,109,0.05)", line_width=0,
                annotation_text="Overbought", annotation_font=dict(color="#FF4D6D", size=8, family="Space Mono"))
            fig_rsi.add_hrect(y0=0, y1=30, fillcolor="rgba(0,245,160,0.05)", line_width=0,
                annotation_text="Oversold", annotation_font=dict(color="#00F5A0", size=8, family="Space Mono"))
            fig_rsi.add_hline(y=50, line=dict(color=GRID_CLR, width=1, dash="dot"))

            l2 = base_layout("RSI (14)")
            l2["height"] = 220
            l2["xaxis"]["range"] = [start_view, end_view]
            l2["xaxis"]["fixedrange"] = False
            l2["yaxis"]["fixedrange"] = True
            l2["yaxis"]["range"] = [0, 100]
            l2["margin"] = dict(l=12, r=12, t=40, b=12)
            fig_rsi.update_layout(**l2)
            st.plotly_chart(fig_rsi, use_container_width=True,
                config={"scrollZoom": True, "responsive": True, "displayModeBar": False})

        with col_macd:
            fig_macd = go.Figure()

            # MACD histogram with opacity gradient
            hist_vals = df["Hist"].fillna(0)
            max_hist = hist_vals.abs().max()
            if max_hist == 0:
                max_hist = 1
            hist_colors = [
                f"rgba(0,245,160,{min(0.9, 0.3 + abs(v)/max_hist*0.7)})" if v >= 0
                else f"rgba(255,77,109,{min(0.9, 0.3 + abs(v)/max_hist*0.7)})"
                for v in hist_vals
            ]

            fig_macd.add_trace(go.Bar(
                x=df["Date"], y=hist_vals,
                marker_color=hist_colors, name="Histogram",
                hovertemplate="Hist: %{y:.4f}<extra></extra>"
            ))
            fig_macd.add_trace(go.Scatter(
                x=df["Date"], y=df["MACD"],
                line=dict(color=BLUE, width=1.5), name="MACD",
                hovertemplate="MACD: %{y:.4f}<extra></extra>"
            ))
            fig_macd.add_trace(go.Scatter(
                x=df["Date"], y=df["Signal"],
                line=dict(color=YELLOW, width=1.5), name="Signal",
                hovertemplate="Signal: %{y:.4f}<extra></extra>"
            ))
            # Calculate visible MACD range to prevent squishing
            macd_min = min(float(df_view["MACD"].min()), float(df_view["Signal"].min()), float(df_view["Hist"].min()))
            macd_max = max(float(df_view["MACD"].max()), float(df_view["Signal"].max()), float(df_view["Hist"].max()))
            macd_pad = (macd_max - macd_min) * 0.05
            if macd_pad == 0:
                macd_pad = abs(macd_min) * 0.05 if macd_min != 0 else 1.0

            l3 = base_layout("MACD (12, 26, 9)")
            l3["height"] = 220
            l3["xaxis"]["range"] = [start_view, end_view]
            l3["xaxis"]["fixedrange"] = False
            l3["yaxis"]["fixedrange"] = True
            l3["yaxis"]["range"] = [macd_min - macd_pad, macd_max + macd_pad]
            l3["margin"] = dict(l=12, r=12, t=40, b=12)
            fig_macd.update_layout(**l3)
            st.plotly_chart(fig_macd, use_container_width=True,
                config={"scrollZoom": True, "responsive": True, "displayModeBar": False})

        st.markdown('<div class="section-header">Volume</div>', unsafe_allow_html=True)
        vol_colors = [GREEN if df["Close"].iloc[i] >= df["Open"].iloc[i]
                      else RED for i in range(len(df))]
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=df["Date"], y=df["Volume"],
            marker_color=vol_colors, opacity=0.7, name="Volume",
            hovertemplate="Vol: %{y:,.0f}<extra></extra>"
        ))
        vol_ma = df["Volume"].rolling(20).mean()
        fig_vol.add_trace(go.Scatter(
            x=df["Date"], y=vol_ma,
            line=dict(color=YELLOW, width=1.5), name="Vol MA 20",
            hovertemplate="Vol MA20: %{y:,.0f}<extra></extra>"
        ))
        # Vol MA annotation
        if not vol_ma.dropna().empty:
            fig_vol.add_annotation(
                x=df["Date"].iloc[-1],
                y=float(vol_ma.iloc[-1]),
                text="  Vol MA20",
                showarrow=False,
                font=dict(color=YELLOW, size=8, family="Space Mono"),
                xanchor="left"
            )
        vol_max = float(df_view["Volume"].max())
        l4 = base_layout("Volume")
        l4["height"] = 180
        l4["xaxis"]["range"] = [start_view, end_view]
        l4["xaxis"]["fixedrange"] = False
        l4["yaxis"]["fixedrange"] = True
        l4["yaxis"]["range"] = [0, vol_max * 1.05]
        l4["margin"] = dict(l=12, r=12, t=40, b=12)
        fig_vol.update_layout(**l4)
        st.plotly_chart(fig_vol, use_container_width=True,
            config={"scrollZoom": True, "responsive": True, "displayModeBar": False})

        # Stock Details
        st.markdown('<div class="section-header">Stock Details</div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        vol_today = float(df["Volume"].iloc[-1])
        vol_avg   = float(df["Volume"].tail(20).mean())
        macd_now  = float(df["MACD"].iloc[-1])
        sig_now   = float(df["Signal"].iloc[-1])
        macd_cross = "🟢 Bullish" if macd_now > sig_now else "🔴 Bearish"

        with d1:
            st.markdown(clean_html(f"""
            <div class="stat-box"><div class="stat-label">Volume Today</div>
            <div class="stat-value">{vol_today/1e6:.2f}M</div></div>
            <div class="stat-box"><div class="stat-label">Avg Volume (20D)</div>
            <div class="stat-value">{vol_avg/1e6:.2f}M</div></div>"""), unsafe_allow_html=True)
        with d2:
            st.markdown(clean_html(f"""
            <div class="stat-box"><div class="stat-label">52W High</div>
            <div class="stat-value bullish">₹{w52_high:,.2f}</div></div>
            <div class="stat-box"><div class="stat-label">52W Low</div>
            <div class="stat-value bearish">₹{w52_low:,.2f}</div></div>"""), unsafe_allow_html=True)
        with d3:
            st.markdown(clean_html(f"""
            <div class="stat-box"><div class="stat-label">MA 50</div>
            <div class="stat-value">₹{float(df['MA50'].iloc[-1]):,.2f}</div></div>
            <div class="stat-box"><div class="stat-label">MA 200</div>
            <div class="stat-value">₹{float(df['MA200'].iloc[-1]):,.2f}</div></div>"""), unsafe_allow_html=True)
        with d4:
            st.markdown(clean_html(f"""
            <div class="stat-box"><div class="stat-label">MACD Signal</div>
            <div class="stat-value" style="font-size:0.85rem;">{macd_cross}</div></div>
            <div class="stat-box"><div class="stat-label">Sector</div>
            <div class="stat-value" style="font-size:0.85rem;color:{SECTOR_COLORS.get(sector,'#5A7499')}">{sector}</div></div>"""),
            unsafe_allow_html=True)

    # ════════════ TAB 2 — 7-DAY FORECAST ════════════
    with tab2:
        if model is None:
            st.error("Model not found.")
        else:
            st.markdown('<div class="section-header">7-Day Price Forecast</div>', unsafe_allow_html=True)
            with st.spinner("Generating 7-day forecast..."):
                forecast_prices = predict_7days(model, scaler, df)
                _, recent_actuals, recent_preds = run_backtest(model, scaler, df, 6)
                recent_metrics = calculate_backtest_metrics(recent_actuals, recent_preds)
                confidence_scores = forecast_confidence_series(recent_metrics["mape"], len(forecast_prices))
            forecast_days = get_trading_days(datetime.today(), 7)
            cols = st.columns(7)
            for i, (col, price, day) in enumerate(zip(cols, forecast_prices, forecast_days)):
                chg   = price - latest_close
                pct   = (chg / latest_close) * 100
                color = "#00F5A0" if chg >= 0 else "#FF4D6D"
                sym   = "▲" if chg >= 0 else "▼"
                conf  = confidence_scores[i]
                with col:
                    st.markdown(clean_html(f"""
                    <div class="forecast-card">
                        <div class="forecast-day">{day.strftime("%a")}<br>{day.strftime("%d %b")}</div>
                        <div class="forecast-price">₹{price:,.0f}</div>
                        <div class="forecast-change" style="color:{color}">{sym} {abs(pct):.1f}%</div>
                        <div style="font-family:Space Mono;font-size:0.55rem;color:#3A5070;margin-top:6px;">~{conf:.0f}% conf.</div>
                    </div>"""), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig_fc = go.Figure()
            last60 = df.tail(60)
            fig_fc.add_trace(go.Scatter(
                x=last60["Date"], y=last60["Close"],
                line=dict(color=BLUE, width=2), name="Actual Price",
                hovertemplate="Close: ₹%{y:,.2f}<extra></extra>"
            ))

            # Backtest-adjusted uncertainty cone.
            returns_std = float(df["Close"].pct_change().tail(60).std())
            error_rate = max(returns_std, recent_metrics["mape"] / 100)
            upper_band = [p * (1 + error_rate * np.sqrt(i+1)) for i, p in enumerate(forecast_prices)]
            lower_band = [p * (1 - error_rate * np.sqrt(i+1)) for i, p in enumerate(forecast_prices)]

            fig_fc.add_trace(go.Scatter(
                x=forecast_days + forecast_days[::-1],
                y=upper_band + lower_band[::-1],
                fill="toself", fillcolor="rgba(0,245,160,0.05)",
                line=dict(color="rgba(255,255,255,0)"), showlegend=False, name="Confidence Band"))

            bridge_dates  = [df["Date"].iloc[-1]] + forecast_days
            bridge_prices = [latest_close] + forecast_prices
            fig_fc.add_trace(go.Scatter(
                x=bridge_dates, y=bridge_prices,
                line=dict(color=GREEN, width=2.5, dash="dot"),
                mode="lines+markers",
                marker=dict(color=GREEN, size=7, symbol="circle"),
                name="7-Day Forecast",
                hovertemplate="Forecast: ₹%{y:,.2f}<extra></extra>"
            ))
            fig_fc.add_vline(x=df["Date"].iloc[-1], line=dict(color="#5A7499", width=1, dash="dash"))
            fig_fc.add_annotation(x=df["Date"].iloc[-1], y=latest_close, text="  Today",
                showarrow=False, font=dict(color="#5A7499", size=9, family="Space Mono"), xanchor="left")

            lf = base_layout(f"7-Day Forecast — {stock_name}")
            lf["height"] = 420
            fig_fc.update_layout(**lf)
            st.plotly_chart(fig_fc, use_container_width=True, config=CHART_CONFIG)
            st.markdown(f"""
            <div class="warn-box">
                ⚠️ Confidence is estimated from the last 6 months of backtest error
                (MAPE: {recent_metrics["mape"]:.2f}%). Multi-day forecasts compound prediction error.
            </div>""", unsafe_allow_html=True)

    # ════════════ TAB 3 — BACKTEST ════════════
    with tab3:
        if model is None:
            st.error("Model not found.")
        else:
            st.markdown('<div class="section-header">Backtest — Predicted vs Actual</div>', unsafe_allow_html=True)
            bt_col1, _ = st.columns([2, 5])
            with bt_col1:
                month_options = {"1 Month":1,"3 Months":3,"6 Months":6,"12 Months":12,"24 Months":24,"36 Months":36}
                bt_range  = st.selectbox("Backtest Period", list(month_options.keys()), index=4)
                bt_months = month_options[bt_range]

            with st.spinner(f"Running backtest for {bt_range}..."):
                bt_dates, bt_actuals, bt_preds = run_backtest(model, scaler, df, bt_months)
                _, _, naive_preds, ma7_preds = run_baseline_backtest(df, bt_months)

            bt_a = np.array(bt_actuals)
            bt_p = np.array(bt_preds)
            lstm_metrics = calculate_backtest_metrics(bt_actuals, bt_preds)
            naive_metrics = calculate_backtest_metrics(bt_actuals, naive_preds)
            ma7_metrics = calculate_backtest_metrics(bt_actuals, ma7_preds)
            mae = lstm_metrics["mae"]
            mape = lstm_metrics["mape"]
            r2 = lstm_metrics["r2"]
            r2c  = "#00F5A0" if r2 > 0.8 else ("#F5C518" if r2 > 0.5 else "#FF4D6D")
            acc  = max(0, min(100, r2 * 100))
            dir_acc = lstm_metrics["direction_accuracy"]

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(clean_html(f"""<div class="stat-box"><div class="stat-label">R² Score</div>
                <div class="stat-value" style="color:{r2c}">{r2:.4f}</div>
                <div class="accuracy-bar-wrap"><div class="accuracy-bar-fill" style="width:{acc:.0f}%"></div></div>
                </div>"""), unsafe_allow_html=True)
            with m2:
                st.markdown(clean_html(f"""<div class="stat-box"><div class="stat-label">MAE</div>
                <div class="stat-value">₹{mae:.2f}</div></div>"""), unsafe_allow_html=True)
            with m3:
                st.markdown(clean_html(f"""<div class="stat-box"><div class="stat-label">MAPE</div>
                <div class="stat-value">{mape:.2f}%</div></div>"""), unsafe_allow_html=True)
            with m4:
                st.markdown(clean_html(f"""<div class="stat-box"><div class="stat-label">Days Tested</div>
                <div class="stat-value">{len(bt_dates)}</div></div>"""), unsafe_allow_html=True)
            with m5:
                st.markdown(clean_html(f"""<div class="stat-box"><div class="stat-label">Direction Accuracy</div>
                <div class="stat-value" style="color:#00F5A0">{dir_acc:.1f}%</div></div>"""), unsafe_allow_html=True)

            st.markdown('<div class="section-header">Baseline Comparison</div>', unsafe_allow_html=True)
            baseline_cols = st.columns(3)
            best_mape = min(mape, naive_metrics["mape"], ma7_metrics["mape"])
            lift_vs_naive = naive_metrics["mape"] - mape
            baseline_cards = [
                ("LSTM", mape, mae, dir_acc),
                ("Naive Close", naive_metrics["mape"], naive_metrics["mae"], naive_metrics["direction_accuracy"]),
                ("7D Average", ma7_metrics["mape"], ma7_metrics["mae"], ma7_metrics["direction_accuracy"]),
            ]
            for col, (label, card_mape, card_mae, card_dir) in zip(baseline_cols, baseline_cards):
                color = "#00F5A0" if card_mape == best_mape else "#5A7499"
                with col:
                    st.markdown(clean_html(f"""
                    <div class="stat-box">
                        <div class="stat-label">{label}</div>
                        <div class="stat-value" style="color:{color}">{card_mape:.2f}%</div>
                        <div style="font-family:Space Mono;font-size:0.62rem;color:#5A7499;margin-top:6px;">
                            MAE ₹{card_mae:.2f} · Dir {card_dir:.1f}%
                        </div>
                    </div>"""), unsafe_allow_html=True)

            lift_color = "#00F5A0" if lift_vs_naive > 0 else "#FF4D6D"
            lift_word = "beats" if lift_vs_naive > 0 else "trails"
            st.markdown(clean_html(f"""
            <div class="warn-box" style="border-color:rgba(0,245,160,0.2);background:rgba(0,245,160,0.03);color:#5A7499;">
                LSTM {lift_word} the naive close baseline by
                <span style="color:{lift_color}">{abs(lift_vs_naive):.2f} MAPE points</span>
                over this period.
            </div>"""), unsafe_allow_html=True)

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=bt_dates, y=bt_actuals,
                line=dict(color=BLUE, width=2), name="Actual Price",
                hovertemplate="Actual: ₹%{y:,.2f}<extra></extra>"
            ))
            fig_bt.add_trace(go.Scatter(
                x=bt_dates, y=bt_preds,
                line=dict(color=GREEN, width=1.5, dash="dot"), name="Predicted Price", opacity=0.9,
                hovertemplate="Predicted: ₹%{y:,.2f}<extra></extra>"
            ))
            fig_bt.add_trace(go.Scatter(
                x=bt_dates+bt_dates[::-1], y=bt_preds+bt_actuals[::-1],
                fill="toself", fillcolor="rgba(0,245,160,0.04)",
                line=dict(color="rgba(255,255,255,0)"), showlegend=False
            ))
            lb = base_layout(f"Backtest — {stock_name} ({bt_range})")
            lb["height"] = 440
            fig_bt.update_layout(**lb)
            st.plotly_chart(fig_bt, use_container_width=True, config=CHART_CONFIG)

            st.markdown('<div class="section-header">Prediction Error Distribution</div>', unsafe_allow_html=True)
            errors = bt_p - bt_a
            fig_err = go.Figure()
            fig_err.add_trace(go.Histogram(
                x=errors, nbinsx=40, marker_color=PURPLE, opacity=0.8, name="Error",
                hovertemplate="Error: ₹%{x:,.2f} · Count: %{y}<extra></extra>"
            ))
            fig_err.add_vline(x=0, line=dict(color=GREEN, width=1.5, dash="dash"))
            le = base_layout("Prediction Error Distribution (₹)")
            le["height"] = 220
            le["margin"] = dict(l=12, r=12, t=40, b=12)
            fig_err.update_layout(**le)
            st.plotly_chart(fig_err, use_container_width=True, config=CHART_CONFIG)

            st.markdown(clean_html(f"""
            <div class="warn-box" style="border-color:rgba(0,245,160,0.2);background:rgba(0,245,160,0.03);color:#5A7499;">
                💡 <span style="color:#00F5A0">Tip:</span> Use <b>24–36 months</b> for the most reliable R² score.
                Current direction accuracy: <span style="color:#00F5A0">{dir_acc:.1f}%</span>
            </div>"""), unsafe_allow_html=True)

    # ════════════ TAB 4 — ABOUT ════════════
    with tab4:
        st.markdown('<div class="section-header">About StockSense India</div>', unsafe_allow_html=True)
        a1, a2 = st.columns(2)

        with a1:
            st.markdown(clean_html("""
            <div class="about-card">
                <div class="about-title">🧠 How It Works</div>
                <div class="about-text">
                    StockSense India uses Long Short-Term Memory (LSTM) neural networks
                    trained on 15+ years of NSE stock data. Each stock has its own
                    dedicated model trained on historical closing prices.<br><br>
                    The model analyzes the last 60 trading days to predict tomorrow's
                    closing price. Technical indicators (RSI, MACD, Bollinger Bands)
                    are calculated separately to generate trading signals.
                </div>
            </div>
            <div class="about-card">
                <div class="about-title">📊 Technical Indicators</div>
                <div class="about-text">
                    <b style="color:#E0E6F0">RSI (14)</b> — Momentum oscillator. Above 70 = overbought, below 30 = oversold.<br><br>
                    <b style="color:#E0E6F0">MACD (12,26,9)</b> — Trend direction and momentum crossovers.<br><br>
                    <b style="color:#E0E6F0">Bollinger Bands (20,2)</b> — Volatility bands that expand/contract with market conditions.<br><br>
                    <b style="color:#E0E6F0">MA 50 & MA 200</b> — Short and long-term trend lines. Golden cross = bullish signal.
                </div>
            </div>"""), unsafe_allow_html=True)

        with a2:
            st.markdown(clean_html('<div class="about-card"><div class="about-title">🏦 Covered Stocks</div><div class="about-text">'),
                        unsafe_allow_html=True)
            for sname, sdata in STOCKS.items():
                sc2 = SECTOR_COLORS.get(sdata['sector'], '#5A7499')
                st.markdown(clean_html(f"""
                <div style="display:flex;justify-content:space-between;padding:5px 0;
                            border-bottom:1px solid #1E2D4A;font-family:Space Mono;font-size:0.65rem;">
                    <span style="color:#E0E6F0">{sname}</span>
                    <span style="color:{sc2}">{sdata['sector']}</span>
                </div>"""), unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

            st.markdown(clean_html("""
            <div class="about-card">
                <div class="about-title">⚙️ Tech Stack</div>
                <div class="about-text">
                    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:4px;">
                        <span style="background:rgba(0,217,245,0.1);border:1px solid rgba(0,217,245,0.3);color:#00D9F5;padding:3px 10px;border-radius:4px;font-size:0.65rem;">Python</span>
                        <span style="background:rgba(0,245,160,0.1);border:1px solid rgba(0,245,160,0.3);color:#00F5A0;padding:3px 10px;border-radius:4px;font-size:0.65rem;">TensorFlow</span>
                        <span style="background:rgba(176,109,255,0.1);border:1px solid rgba(176,109,255,0.3);color:#B06DFF;padding:3px 10px;border-radius:4px;font-size:0.65rem;">Keras LSTM</span>
                        <span style="background:rgba(245,197,24,0.1);border:1px solid rgba(245,197,24,0.3);color:#F5C518;padding:3px 10px;border-radius:4px;font-size:0.65rem;">Streamlit</span>
                        <span style="background:rgba(0,217,245,0.1);border:1px solid rgba(0,217,245,0.3);color:#00D9F5;padding:3px 10px;border-radius:4px;font-size:0.65rem;">Plotly</span>
                        <span style="background:rgba(255,77,109,0.1);border:1px solid rgba(255,77,109,0.3);color:#FF4D6D;padding:3px 10px;border-radius:4px;font-size:0.65rem;">yfinance</span>
                        <span style="background:rgba(0,245,160,0.1);border:1px solid rgba(0,245,160,0.3);color:#00F5A0;padding:3px 10px;border-radius:4px;font-size:0.65rem;">scikit-learn</span>
                    </div>
                </div>
            </div>"""), unsafe_allow_html=True)

        st.markdown(clean_html("""
        <div class="warn-box" style="margin-top:16px;">
            ⚠️ <b>Disclaimer:</b> StockSense India is built purely for educational purposes.
            Predictions should NOT be used as financial advice. Always consult a SEBI-registered
            financial advisor before making investment decisions.
        </div>"""), unsafe_allow_html=True)

    # ─── FOOTER ──────────────────────────────────────────────────────────────────────
    st.markdown(clean_html("""
    <div class="footer">
        <div class="footer-logo">StockSense India</div>
        <div class="footer-text">Built with LSTM · TensorFlow · Streamlit · Data via yfinance</div>
        <div class="footer-links">
            <a href="#">GitHub</a> · <a href="#">About</a> · <a href="#">Disclaimer</a>
        </div>
        <div class="footer-text" style="margin-top:8px;">© 2026 · For educational use only · Not financial advice</div>
    </div>
    """), unsafe_allow_html=True)
