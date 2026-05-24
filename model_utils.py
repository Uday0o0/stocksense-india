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
