# app.py -- quick prototype using yfinance
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Bloomberg-Lite")
st.title("Bloomberg-Lite — Prototype")

# Sidebar
st.sidebar.markdown("## Watchlist")
default_watch = ["AAPL", "TSLA", "MSFT", "NSE:RELIANCE"]  # mix for testing
watch = st.sidebar.text_area("Tickers (comma separated)", value=",".join(default_watch))
tickers = [t.strip().upper() for t in watch.split(",") if t.strip()]

# Ticker selector
ticker = st.selectbox("Choose ticker", options=tickers)

# Fetch data
@st.cache_data(ttl=60)  # cache for 60s during prototyping
def fetch_history(sym, period="1mo", interval="1d"):
    try:
        t = yf.Ticker(sym)
        hist = t.history(period=period, interval=interval)
        return hist
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

hist = fetch_history(ticker, period="1mo", interval="1d")

# Price card
if not hist.empty:
    last_close = hist["Close"].iloc[-1]
    prev = hist["Close"].iloc[-2] if len(hist) > 1 else last_close
    change = last_close - prev
    pct = (change / prev) * 100 if prev else 0
    st.metric(label=f"{ticker} price", value=f"{last_close:.2f}", delta=f"{pct:.2f}%")

# Chart
st.line_chart(hist["Close"].rename("Close"))

# Fundamentals (simple)
st.header("Fundamentals / Info")
info = yf.Ticker(ticker).info
funds = {
    "marketCap": info.get("marketCap"),
    "trailingPE": info.get("trailingPE"),
    "forwardPE": info.get("forwardPE"),
    "dividendYield": info.get("dividendYield"),
}
st.json(funds)

# News (from yfinance news if available)
st.header("News")
news = yf.Ticker(ticker).news
for n in news[:5]:
    st.write(f"**{n.get('title')}** — {n.get('publisher')}")
    st.write(n.get('link'))
