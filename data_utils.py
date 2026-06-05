from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=300)
def load_data(ticker):
    for attempt in range(3):
        try:
            data = yf.download(
                ticker,
                start="2010-01-01",
                end=datetime.today().strftime("%Y-%m-%d"),
                progress=False,
                timeout=30,
                auto_adjust=True,
            )
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.index.name = "Date"
            data = data.reset_index()
            if "Date" not in data.columns:
                data["Date"] = data.index
            data["Date"] = pd.to_datetime(data["Date"])
            return data
        except Exception:
            if attempt == 2:
                return None
    return None


@st.cache_data(ttl=600)
def load_index_data():
    indices = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
    result = {}
    for name, ticker in indices.items():
        try:
            data = yf.download(ticker, period="5d", progress=False, timeout=15)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data = data.reset_index()
                if len(data) >= 2:
                    curr = float(data["Close"].iloc[-1])
                    prev = float(data["Close"].iloc[-2])
                    chg = curr - prev
                    pct = (chg / prev) * 100
                    result[name] = {"price": curr, "change": chg, "pct": pct}
        except Exception:
            result[name] = {"price": 0, "change": 0, "pct": 0}
    return result
