import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import os
import onnxruntime as ort
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense India",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00F5A0, #00D9F5);
}
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
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ───────────────────────────────────────────────────────────────────
STOCKS = {
    "Reliance Industries": {"ticker": "RELIANCE.NS",   "model": "Reliance"},
    "TCS":                 {"ticker": "TCS.NS",         "model": "TCS"},
    "Infosys":             {"ticker": "INFY.NS",        "model": "Infosys"},
    "HDFC Bank":           {"ticker": "HDFCBANK.NS",    "model": "HDFC_Bank"},
    "ICICI Bank":          {"ticker": "ICICIBANK.NS",   "model": "ICICI_Bank"},
    "Wipro":               {"ticker": "WIPRO.NS",       "model": "Wipro"},
    "Bajaj Finance":       {"ticker": "BAJFINANCE.NS",  "model": "Bajaj_Finance"},
    "Bharti Airtel":       {"ticker": "BHARTIARTL.NS",  "model": "Bharti_Airtel"},
    "Larsen & Toubro":     {"ticker": "LT.NS",          "model": "LT"},
    "Asian Paints":        {"ticker": "ASIANPAINT.NS",  "model": "Asian_Paints"},
}

SAVE_DIR = os.path.dirname(__file__)
LOOKBACK = 60

# ─── HELPERS ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
@st.cache_data(ttl=300)
def load_data(ticker):
    for attempt in range(3):
        try:
            data = yf.download(ticker, start="2010-01-01",
                               end=datetime.today().strftime("%Y-%m-%d"),
                               progress=False, timeout=30)
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.reset_index(inplace=True)
            return data
        except Exception:
            if attempt == 2:
                return None
    return None

@st.cache_resource
def load_onnx_session(model_name):
    onnx_path = os.path.join(SAVE_DIR, f"{model_name}.onnx")
    if not os.path.exists(onnx_path):
        return None
    return ort.InferenceSession(onnx_path)

@st.cache_resource
def load_scaler(model_name):
    scaler_path = os.path.join(SAVE_DIR, f"{model_name}_scaler.pkl")
    if not os.path.exists(scaler_path):
        return None
    return pickle.load(open(scaler_path, "rb"))

def calc_rsi(source, length=14):
    change   = pd.Series(source).diff()
    up       = change.clip(lower=0)
    down     = -change.clip(upper=0)
    alpha    = 1 / length
    up_rma   = up.ewm(alpha=alpha, adjust=False).mean()
    down_rma = down.ewm(alpha=alpha, adjust=False).mean()
    rsi = np.where(down_rma == 0, 100,
          np.where(up_rma   == 0,   0,
          100 - (100 / (1 + up_rma / down_rma))))
    return pd.Series(rsi, index=pd.Series(source).index)

def add_indicators(df):
    df = df.copy()
    c = df["Close"]
    df["MA50"]     = c.rolling(50).mean()
    df["MA200"]    = c.rolling(200).mean()
    df["RSI"]      = calc_rsi(c)
    ema12          = c.ewm(span=12, adjust=False).mean()
    ema26          = c.ewm(span=26, adjust=False).mean()
    df["MACD"]     = ema12 - ema26
    df["Signal"]   = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"]     = df["MACD"] - df["Signal"]
    bb_mid         = c.rolling(20).mean()
    bb_std         = c.rolling(20).std()
    df["BB_Mid"]   = bb_mid
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std
    return df

def get_signal(df):
    latest = df.iloc[-1]
    rsi    = latest["RSI"]
    ma50   = latest["MA50"]
    ma200  = latest["MA200"]
    close  = latest["Close"]
    score  = 0
    if rsi < 35:   score += 2
    elif rsi > 70: score -= 2
    if close > ma50:  score += 1
    if ma50 > ma200:  score += 1
    if close < ma50:  score -= 1
    if ma50 < ma200:  score -= 1
    if   score >= 2:  return "BUY",  "badge-buy",  "bullish"
    elif score <= -2: return "SELL", "badge-sell", "bearish"
    else:             return "HOLD", "badge-hold", "neutral"

def next_trading_day(from_date):
    d = from_date + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d

def predict_tomorrow(session, scaler, df):
    close  = df["Close"].to_numpy().flatten().reshape(-1, 1)
    scaled = scaler.transform(close)
    seq    = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1).astype(np.float32)
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    pred   = session.run([output_name], {input_name: seq})[0]
    price  = scaler.inverse_transform(pred)[0][0]
    return float(price)

# ─── PLOTLY THEME ────────────────────────────────────────────────────────────────
PLOT_BG  = "#090E1A"
GRID_CLR = "#1E2D4A"
TEXT_CLR = "#5A7499"
GREEN    = "#00F5A0"
RED      = "#FF4D6D"
BLUE     = "#00D9F5"
YELLOW   = "#F5C518"

def base_layout(title=""):
    return dict(
        title=dict(text=title, font=dict(family="Syne", size=13, color="#E0E6F0")),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Space Mono", color=TEXT_CLR, size=10),
        xaxis=dict(gridcolor=GRID_CLR, showgrid=True, zeroline=False,
                   tickfont=dict(size=9), rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor=GRID_CLR, showgrid=True, zeroline=False,
                   tickfont=dict(size=9)),
        margin=dict(l=12, r=12, t=36, b=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        hovermode="x unified",
    )

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo-header">StockSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">India · AI Predictions</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Select Stock</div>', unsafe_allow_html=True)
    stock_name = st.selectbox("", list(STOCKS.keys()), label_visibility="collapsed")

    st.markdown('<div class="section-header">Chart Range</div>', unsafe_allow_html=True)
    range_option = st.selectbox("", ["3 Months", "6 Months", "1 Year", "2 Years", "Max"],
                                label_visibility="collapsed")

    range_map = {"3 Months": 90, "6 Months": 180, "1 Year": 365, "2 Years": 730, "Max": 9999}
    days_back = range_map[range_option]

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="warn-box">⚠️ For educational use only. Not financial advice.</div>',
        unsafe_allow_html=True
    )

# ─── STOCK INFO ──────────────────────────────────────────────────────────────────
ticker     = STOCKS[stock_name]["ticker"]
model_name = STOCKS[stock_name]["model"]

col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown(
        f"## {stock_name} &nbsp;"
        f"<span style='font-size:0.8rem;color:#3A5070;font-family:Space Mono'>{ticker}</span>"
        f"<span class='model-badge'>LSTM · ONNX</span>",
        unsafe_allow_html=True
    )
with col_refresh:
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("↻ Refresh", use_container_width=True)

# ─── LOAD DATA ───────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching {stock_name} data..."):
    df = load_data(ticker)

if df is None or df.empty:
    st.error("Could not load data. Please try again.")
    st.stop()

df = add_indicators(df)

if days_back < 9999:
    cutoff  = pd.Timestamp(datetime.today() - timedelta(days=days_back))
    df_view = df[df["Date"] >= cutoff].copy()
else:
    df_view = df.copy()

latest_close = float(df["Close"].iloc[-1])
prev_close   = float(df["Close"].iloc[-2])
day_change   = latest_close - prev_close
day_pct      = (day_change / prev_close) * 100
chg_class    = "bullish" if day_change >= 0 else "bearish"
chg_symbol   = "▲" if day_change >= 0 else "▼"

# ─── PREDICTION ──────────────────────────────────────────────────────────────────
with st.spinner(f"Running LSTM prediction for {stock_name}..."):
    session = load_onnx_session(model_name)
    scaler  = load_scaler(model_name)

pred_price  = None
pred_change = None
pred_pct    = None
pred_day    = None

if session is not None and scaler is not None:
    pred_price  = predict_tomorrow(session, scaler, df)
    pred_change = pred_price - latest_close
    pred_pct    = (pred_change / latest_close) * 100
    pred_day    = next_trading_day(datetime.today())

signal, badge_class, signal_class = get_signal(df)

# ─── KPI ROW ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Price</div>
        <div class="metric-value">₹{latest_close:,.2f}</div>
        <div class="metric-sub {chg_class}">{chg_symbol} ₹{abs(day_change):.2f} ({abs(day_pct):.2f}%) today</div>
    </div>""", unsafe_allow_html=True)

with c2:
    if pred_price:
        p_class  = "bullish" if pred_change >= 0 else "bearish"
        p_symbol = "▲" if pred_change >= 0 else "▼"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tomorrow's Prediction</div>
            <div class="metric-value">₹{pred_price:,.2f}</div>
            <div class="metric-sub {p_class}">{p_symbol} ₹{abs(pred_change):.2f} ({abs(pred_pct):.2f}%) · {pred_day.strftime("%d %b")}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tomorrow's Prediction</div>
            <div class="metric-value" style="font-size:1rem;color:#3A5070;">Model not found</div>
            <div class="metric-sub neutral">ONNX file missing</div>
        </div>""", unsafe_allow_html=True)

with c3:
    rsi_val   = float(df["RSI"].iloc[-1])
    rsi_class = "bullish" if rsi_val < 40 else ("bearish" if rsi_val > 65 else "neutral")
    rsi_lbl   = "Oversold" if rsi_val < 30 else ("Overbought" if rsi_val > 70 else "Neutral")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RSI (14)</div>
        <div class="metric-value">{rsi_val:.1f}</div>
        <div class="metric-sub {rsi_class}">{rsi_lbl}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Signal</div>
        <div class="metric-value" style="font-size:1.1rem;padding-top:6px;">
            <span class="signal-badge {badge_class}">{signal}</span>
        </div>
        <div class="metric-sub" style="margin-top:10px;color:#3A5070;">RSI + MA cross</div>
    </div>""", unsafe_allow_html=True)

# ─── PRICE CHART ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Price Chart · Bollinger Bands · Moving Averages</div>', unsafe_allow_html=True)

fig_price = go.Figure()
fig_price.add_trace(go.Candlestick(
    x=df_view["Date"],
    open=df_view["Open"], high=df_view["High"],
    low=df_view["Low"],   close=df_view["Close"],
    increasing_line_color=GREEN, decreasing_line_color=RED,
    increasing_fillcolor=GREEN,  decreasing_fillcolor=RED,
    name="OHLC", opacity=0.85
))
fig_price.add_trace(go.Scatter(
    x=df_view["Date"], y=df_view["BB_Upper"],
    line=dict(color="rgba(0,217,245,0.3)", width=1, dash="dot"), name="BB Upper"
))
fig_price.add_trace(go.Scatter(
    x=df_view["Date"], y=df_view["BB_Lower"],
    fill="tonexty", fillcolor="rgba(0,217,245,0.04)",
    line=dict(color="rgba(0,217,245,0.3)", width=1, dash="dot"), name="BB Lower"
))
fig_price.add_trace(go.Scatter(
    x=df_view["Date"], y=df_view["BB_Mid"],
    line=dict(color="rgba(0,217,245,0.5)", width=1), name="BB Mid"
))
fig_price.add_trace(go.Scatter(
    x=df_view["Date"], y=df_view["MA50"],
    line=dict(color=YELLOW, width=1.5), name="MA 50"
))
fig_price.add_trace(go.Scatter(
    x=df_view["Date"], y=df_view["MA200"],
    line=dict(color="#FF6B9D", width=1.5), name="MA 200"
))
if pred_price:
    fig_price.add_trace(go.Scatter(
        x=[pred_day], y=[pred_price],
        mode="markers+text",
        marker=dict(color=GREEN, size=12, symbol="star",
                    line=dict(color="white", width=1.5)),
        text=[f"  ₹{pred_price:,.0f}"],
        textfont=dict(color=GREEN, size=10, family="Space Mono"),
        textposition="middle right",
        name="LSTM Prediction"
    ))

l = base_layout()
l["height"] = 440
fig_price.update_layout(**l)
st.plotly_chart(fig_price, use_container_width=True)

# ─── INDICATORS ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Indicators</div>', unsafe_allow_html=True)
col_rsi, col_macd = st.columns(2)

with col_rsi:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=df_view["Date"], y=df_view["RSI"],
        line=dict(color=BLUE, width=1.8),
        fill="tozeroy", fillcolor="rgba(0,217,245,0.05)", name="RSI"
    ))
    fig_rsi.add_hline(y=70, line=dict(color=RED,      width=1, dash="dot"))
    fig_rsi.add_hline(y=30, line=dict(color=GREEN,    width=1, dash="dot"))
    fig_rsi.add_hline(y=50, line=dict(color=GRID_CLR, width=1, dash="dot"))
    l2 = base_layout("RSI (14)")
    l2["height"] = 200
    l2["yaxis"]["range"] = [0, 100]
    l2["margin"] = dict(l=12, r=12, t=30, b=12)
    fig_rsi.update_layout(**l2)
    st.plotly_chart(fig_rsi, use_container_width=True)

with col_macd:
    fig_macd = go.Figure()
    colors_hist = [GREEN if v >= 0 else RED for v in df_view["Hist"].fillna(0)]
    fig_macd.add_trace(go.Bar(
        x=df_view["Date"], y=df_view["Hist"],
        marker_color=colors_hist, name="Histogram", opacity=0.7
    ))
    fig_macd.add_trace(go.Scatter(
        x=df_view["Date"], y=df_view["MACD"],
        line=dict(color=BLUE, width=1.5), name="MACD"
    ))
    fig_macd.add_trace(go.Scatter(
        x=df_view["Date"], y=df_view["Signal"],
        line=dict(color=YELLOW, width=1.5), name="Signal"
    ))
    l3 = base_layout("MACD (12, 26, 9)")
    l3["height"] = 200
    l3["margin"] = dict(l=12, r=12, t=30, b=12)
    fig_macd.update_layout(**l3)
    st.plotly_chart(fig_macd, use_container_width=True)

# ─── VOLUME ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Volume</div>', unsafe_allow_html=True)
vol_colors = [GREEN if df_view["Close"].iloc[i] >= df_view["Open"].iloc[i]
              else RED for i in range(len(df_view))]
fig_vol = go.Figure()
fig_vol.add_trace(go.Bar(
    x=df_view["Date"], y=df_view["Volume"],
    marker_color=vol_colors, opacity=0.7, name="Volume"
))
l4 = base_layout("Volume")
l4["height"] = 160
l4["margin"] = dict(l=12, r=12, t=30, b=12)
fig_vol.update_layout(**l4)
st.plotly_chart(fig_vol, use_container_width=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;font-family:Space Mono;font-size:0.62rem;color:#1E2D4A;">'
    'StockSense India · LSTM · ONNX Runtime · Data via yfinance · For educational use only'
    '</div>',
    unsafe_allow_html=True
)
