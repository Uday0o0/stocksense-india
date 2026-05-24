import os


STOCKS = {
    "Reliance Industries": {"ticker": "RELIANCE.NS", "model": "Reliance", "sector": "Energy"},
    "TCS": {"ticker": "TCS.NS", "model": "TCS", "sector": "IT"},
    "Infosys": {"ticker": "INFY.NS", "model": "Infosys", "sector": "IT"},
    "HDFC Bank": {"ticker": "HDFCBANK.NS", "model": "HDFC_Bank", "sector": "Banking"},
    "ICICI Bank": {"ticker": "ICICIBANK.NS", "model": "ICICI_Bank", "sector": "Banking"},
    "Wipro": {"ticker": "WIPRO.NS", "model": "Wipro", "sector": "IT"},
    "Bajaj Finance": {"ticker": "BAJFINANCE.NS", "model": "Bajaj_Finance", "sector": "NBFC"},
    "Bharti Airtel": {"ticker": "BHARTIARTL.NS", "model": "Bharti_Airtel", "sector": "Telecom"},
    "Larsen & Toubro": {"ticker": "LT.NS", "model": "LT", "sector": "Infrastructure"},
    "Asian Paints": {"ticker": "ASIANPAINT.NS", "model": "Asian_Paints", "sector": "Consumer"},
}

SECTOR_COLORS = {
    "Energy": "#FF8C42",
    "IT": "#00D9F5",
    "Banking": "#B06DFF",
    "NBFC": "#F5C518",
    "Telecom": "#00F5A0",
    "Infrastructure": "#FF6B9D",
    "Consumer": "#7EB8F5",
}

SAVE_DIR = os.path.dirname(__file__)
LOOKBACK = 60
