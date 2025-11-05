import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime

st.set_page_config(page_title="Bloomberg-Lite (Easy Mode)", layout="wide")
st.title("Bloomberg-Lite — Easy Mode (No Docker)")

# ---- Watchlist ----
watchlist = st.sidebar.text_area("Tickers (comma separated)", "AAPL,MSFT,TSLA,RELIANCE.NS")
tickers = [t.strip().upper() for t in watchlist.split(",") if t.strip()]
refresh = st.sidebar.slider("Refresh every (seconds)", 5, 60, 10)
auto = st.sidebar.checkbox("Auto-refresh data", True)

st.sidebar.markdown("**Tip:** Works perfectly on Streamlit Cloud — no Docker required.")

# ---- Fetch data ----
@st.cache_data(ttl=30)
def get_data(ticker):
    t = yf.Ticker(ticker)
    hist = t.history(period="1mo")
    last = hist["Close"].iloc[-1]
    prev = hist["Close"].iloc[-2]
    delta = last - prev
    pct = (delta / prev) * 100
    return {"last": round(last, 2), "delta": round(delta, 2), "pct": round(pct, 2), "history": hist}

# ---- Dashboard loop ----
placeholder = st.empty()

def render_dashboard():
    with placeholder.container():
        st.subheader(f"Market Snapshot ({datetime.now().strftime('%H:%M:%S')})")
        cols = st.columns(4)
        for i, t in enumerate(tickers):
            data = get_data(t)
            color = "🟢" if data["pct"] > 0 else "🔴"
            cols[i % 4].metric(label=t, value=f"{data['last']} {color}", delta=f"{data['pct']}%")
        st.markdown("---")
        sel = st.selectbox("View chart for:", tickers)
        df = get_data(sel)["history"]
        st.line_chart(df["Close"])
        st.caption("Live simulation: refreshes every few seconds for demo effect.")

# ---- Run loop ----
if auto:
    while True:
        render_dashboard()
        time.sleep(refresh)
else:
    render_dashboard()
