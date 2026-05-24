from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def calc_rsi(source, length=14):
    change = pd.Series(source).diff()
    up = change.clip(lower=0)
    down = -change.clip(upper=0)
    alpha = 1 / length
    up_rma = up.ewm(alpha=alpha, adjust=False).mean()
    down_rma = down.ewm(alpha=alpha, adjust=False).mean()
    rsi = np.where(
        down_rma == 0,
        100,
        np.where(up_rma == 0, 0, 100 - (100 / (1 + up_rma / down_rma))),
    )
    return pd.Series(rsi, index=pd.Series(source).index)


def add_indicators(df):
    df = df.copy()
    close = df["Close"]
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()
    df["RSI"] = calc_rsi(close)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"] = df["MACD"] - df["Signal"]

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_Mid"] = bb_mid
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std
    return df


def get_signal(df):
    latest = df.iloc[-1]
    rsi = latest["RSI"]
    ma50 = latest["MA50"]
    ma200 = latest["MA200"]
    close = latest["Close"]
    score = 0

    if rsi < 35:
        score += 2
    elif rsi > 70:
        score -= 2
    if close > ma50:
        score += 1
    if ma50 > ma200:
        score += 1
    if close < ma50:
        score -= 1
    if ma50 < ma200:
        score -= 1

    if score >= 2:
        return "BUY", "badge-buy", "bullish"
    if score <= -2:
        return "SELL", "badge-sell", "bearish"
    return "HOLD", "badge-hold", "neutral"


def get_52w(df):
    one_year_ago = pd.Timestamp(datetime.today() - timedelta(days=365))
    df_1y = df[df["Date"] >= one_year_ago]
    if df_1y.empty:
        return None, None
    return float(df_1y["Low"].min()), float(df_1y["High"].max())
