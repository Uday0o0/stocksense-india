import os
import pickle
from datetime import timedelta

import numpy as np
import streamlit as st
from keras.models import load_model

from constants import LOOKBACK, SAVE_DIR


@st.cache_resource
def load_model_and_scaler(model_name):
    model_path = os.path.join(SAVE_DIR, f"{model_name}_model.keras")
    scaler_path = os.path.join(SAVE_DIR, f"{model_name}_scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    model = load_model(model_path)
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler


def next_trading_day(from_date):
    day = from_date + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def get_trading_days(from_date, n):
    days = []
    day = from_date
    while len(days) < n:
        day = day + timedelta(days=1)
        if day.weekday() < 5:
            days.append(day)
    return days


def predict_one(model, scaler, sequence):
    seq = sequence[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    pred = model.predict(seq, verbose=0)
    return float(scaler.inverse_transform(pred)[0][0])


def predict_tomorrow(model, scaler, df):
    close = df["Close"].to_numpy().flatten().reshape(-1, 1)
    scaled = scaler.transform(close)
    return predict_one(model, scaler, scaled)


def predict_7days(model, scaler, df):
    close = df["Close"].to_numpy().flatten().reshape(-1, 1)
    scaled = scaler.transform(close).flatten().tolist()
    preds = []
    seq = scaled.copy()

    for _ in range(7):
        inp = np.array(seq[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        pred = model.predict(inp, verbose=0)
        price = float(scaler.inverse_transform(pred)[0][0])
        seq.append(float(scaler.transform([[price]])[0][0]))
        preds.append(price)
    return preds


def run_backtest(model, scaler, df, months):
    close = df["Close"].to_numpy().flatten().reshape(-1, 1)
    scaled = scaler.transform(close).flatten()
    days = int(months * 21)
    start_i = max(LOOKBACK, len(scaled) - days - 1)
    actuals, preds, dates = [], [], []

    for i in range(start_i, len(scaled)):
        seq = scaled[i - LOOKBACK:i].reshape(1, LOOKBACK, 1)
        pred = model.predict(seq, verbose=0)
        preds.append(float(scaler.inverse_transform(pred)[0][0]))
        actuals.append(float(close[i][0]))
        dates.append(df["Date"].iloc[i])
    return dates, actuals, preds


def run_baseline_backtest(df, months):
    close = df["Close"].to_numpy().flatten()
    days = int(months * 21)
    start_i = max(LOOKBACK, len(close) - days - 1)
    actuals, naive_preds, ma7_preds, dates = [], [], [], []

    for i in range(start_i, len(close)):
        actuals.append(float(close[i]))
        naive_preds.append(float(close[i - 1]))
        ma7_preds.append(float(np.mean(close[max(0, i - 7):i])))
        dates.append(df["Date"].iloc[i])
    return dates, actuals, naive_preds, ma7_preds


def calculate_backtest_metrics(actuals, preds):
    actuals = np.asarray(actuals, dtype=float)
    preds = np.asarray(preds, dtype=float)
    errors = actuals - preds

    mae = float(np.mean(np.abs(errors)))
    nonzero_actuals = np.where(actuals == 0, np.nan, actuals)
    mape = float(np.nanmean(np.abs(errors / nonzero_actuals)) * 100)

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((actuals - np.mean(actuals)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot else 0.0

    if len(actuals) > 1:
        direction_accuracy = sum(
            1
            for i in range(1, len(actuals))
            if (preds[i] > preds[i - 1]) == (actuals[i] > actuals[i - 1])
        ) / (len(actuals) - 1) * 100
    else:
        direction_accuracy = 0.0

    return {
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "direction_accuracy": float(direction_accuracy),
        "errors": preds - actuals,
    }


def forecast_confidence_series(backtest_mape, horizon=7):
    base_confidence = max(35, min(90, 100 - backtest_mape * 5))
    daily_decay = max(4, min(10, backtest_mape))
    return [max(30, base_confidence - i * daily_decay) for i in range(horizon)]
