import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
import time
from datetime import datetime

# Optional auto-refresh helper (install streamlit-autorefresh to enable)
try:
    from streamlit_autorefresh import st_autorefresh
    AUTORELOAD_AVAILABLE = True
except Exception:
    AUTORELOAD_AVAILABLE = False

# Optional OpenAI (new API syntax)
try:
    from openai import OpenAI
    OPENAI_INSTALLED = True
except Exception:
    OPENAI_INSTALLED = False

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Bloomberg-Lite AI News App", layout="wide", initial_sidebar_state="expanded")
st.title("📡 Bloomberg-Lite — AI-Powered News & Market Dashboard")

# ---------- SESSION STATE ----------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AAPL", "MSFT", "TSLA", "RELIANCE.NS"]
if "positions" not in st.session_state:
    st.session_state.positions = []
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "polling" not in st.session_state:
    st.session_state.polling = False
if "live_prices" not in st.session_state:
    st.session_state.live_prices = {}  # symbol -> {price, ts}

# ---------- UTILITIES ----------
@st.cache_data(ttl=300)
def get_history(sym, period="1y", interval="1d"):
    """Return yfinance history as DataFrame or empty DataFrame on failure."""
    sym = (sym or "").strip()
    if not sym:
        return pd.DataFrame()
    try:
        df = yf.Ticker(sym).history(period=period, interval=interval)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        return df
    except Exception:
        return pd.DataFrame()

def get_latest_price(sym):
    """Fetch latest intraday price (close of last 1m candle) — returns float or None."""
    sym = (sym or "").strip()
    if not sym:
        return None
    try:
        # Use 1m interval for freshness (yfinance may return empty for non-US markets or if market closed)
        df = yf.Ticker(sym).history(period="1d", interval="1m")
        if df is None or df.empty or "Close" not in df.columns:
            # fallback: try 5d 1d candle
            df2 = yf.Ticker(sym).history(period="5d", interval="1d")
            if df2 is None or df2.empty or "Close" not in df2.columns:
                return None
            return float(df2["Close"].iloc[-1])
        return float(df["Close"].iloc[-1])
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_finance_news(api_key, query="finance OR investing", language="en", page_size=10):
    if not api_key:
        return []
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={requests.utils.quote(query)}&language={language}&pageSize={page_size}&sortBy=publishedAt&apiKey={api_key}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        articles = j.get("articles", [])
        results = []
        for a in articles:
            results.append({
                "title": a.get("title"),
                "source": a.get("source", {}).get("name", ""),
                "publishedAt": a.get("publishedAt", ""),
                "url": a.get("url", ""),
                "description": a.get("description", "")
            })
        return results
    except Exception:
        return []

# ---------- LAYOUT ----------
tabs = st.tabs(["Market", "News Hub", "Portfolio", "Alerts", "Reports"])

# ---------- MARKET TAB (with Live Polling) ----------
with tabs[0]:
    st.header("Market Overview — Live Polling (safe)")
    st.markdown("""
    - Use the controls below to start/stop live polling for the selected watchlist.
    - Default interval: 15 seconds. Keep intervals >= 10s to avoid rate limits.
    """)

    # watchlist editor
    wl = st.text_area("Watchlist (comma separated)", ",".join(st.session_state.watchlist), height=80)
    if st.button("Update Watchlist"):
        st.session_state.watchlist = [s.strip().upper() for s in wl.split(",") if s.strip()]
        st.success("Watchlist updated.")

    # Poll controls
    col_a, col_b, col_c = st.columns([2,1,1])
    with col_a:
        sel = st.selectbox("Select ticker to view", st.session_state.watchlist)
    with col_b:
        interval = st.number_input("Poll interval (secs)", min_value=5, value=15, step=1, help="How often to poll prices (>=10 recommended)")
    with col_c:
        if st.session_state.polling:
            if st.button("Stop Live Polling"):
                st.session_state.polling = False
                st.success("Stopped live polling.")
        else:
            if st.button("Start Live Polling"):
                st.session_state.polling = True
                st.success("Started live polling (UI will refresh).")

    # Optional auto-refresh (recommended)
    if st.session_state.polling and AUTORELOAD_AVAILABLE:
        # st_autorefresh reruns the app every `interval*1000` ms
        st_autorefresh(interval=int(interval*1000), key="autorefresh")

    # Fetch latest price(s) to display
    # If polling is enabled we update session_state.live_prices for all watchlist symbols
    placeholder = st.empty()
    if st.session_state.polling:
        # Update each symbol's latest price (rate-limited by interval and streamlit-autorefresh)
        for s in st.session_state.watchlist:
            price = get_latest_price(s)
            if price is not None:
                st.session_state.live_prices[s] = {"price": price, "ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
    else:
        # Not polling: show last known prices if available
        pass

    # Display a simple table of live prices
    if st.session_state.live_prices:
        rows = []
        for s in st.session_state.watchlist:
            info = st.session_state.live_prices.get(s)
            if info:
                rows.append({"symbol": s, "last_price": info["price"], "updated_at": info["ts"]})
            else:
                rows.append({"symbol": s, "last_price": None, "updated_at": None})
        df_live = pd.DataFrame(rows)
        placeholder.dataframe(df_live)
    else:
        placeholder.info("No live prices yet. Start polling to fetch latest quotes, or click 'Refresh prices' below.")

    # Manual refresh button (fallback if you don't have streamlit-autorefresh)
    if st.button("Refresh prices now"):
        # update only selected ticker for speed
        p = get_latest_price(sel)
        if p is not None:
            st.session_state.live_prices[sel] = {"price": p, "ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
            st.success(f"{sel} — {p}")
        else:
            st.error(f"Could not fetch latest price for {sel}.")

    # show chart for selected ticker (1d daily)
    hist = get_history(sel)
    if not hist.empty and "Close" in hist.columns:
        if "Date" in hist.columns:
            ch = hist.set_index("Date")["Close"]
        else:
            ch = hist["Close"]
        st.line_chart(ch)
    else:
        st.info("No historical data to show for selected ticker.")

# ---------- NEWS HUB TAB ----------
with tabs[1]:
    st.header("📰 Finance & Investing News")
    api_key = st.secrets.get("NEWSAPI_KEY") if hasattr(st, "secrets") else None
    if api_key:
        with st.spinner("Fetching news..."):
            news_list = fetch_finance_news(api_key)
        if news_list:
            for n in news_list:
                st.markdown(f"**{n['title']}**  \n*{n['source']} — {n['publishedAt'][:10]}*  \n[{n['url']}]({n['url']})")
                if n["description"]:
                    st.markdown(f"_{n['description']}_")
                st.markdown("---")
        else:
            st.info("No recent finance news found.")
        if OPENAI_INSTALLED and "OPENAI_API_KEY" in st.secrets:
            if st.button("🧠 Generate AI Summary of Headlines"):
                titles = "\n".join([n["title"] for n in news_list[:8]])
                try:
                    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a financial journalist summarizing key market headlines."},
                            {"role": "user", "content": f"Summarize these finance headlines into 3 concise bullet points:\n{titles}"}
                        ],
                        max_tokens=200,
                        temperature=0.4,
                    )
                    summary = response.choices[0].message.content
                    st.subheader("🧠 AI Summary")
                    st.write(summary)
                except Exception as e:
                    st.error(f"OpenAI summarization failed: {e}")
        else:
            st.info("Add OPENAI_API_KEY to enable AI summaries.")
    else:
        st.warning("Add NEWSAPI_KEY in Streamlit secrets to fetch live finance news.")

# ---------- PORTFOLIO TAB ----------
with tabs[2]:
    st.header("Portfolio Simulator")
    sym = st.text_input("Symbol to add")
    qty = st.number_input("Qty", min_value=1, value=1)
    side = st.selectbox("Side", ["long", "short"])
    if st.button("Add Position"):
        hist = get_history(sym, period="5d")
        price = hist["Close"].iloc[-1] if not hist.empty else 0.0
        st.session_state.positions.append({"symbol": sym.upper(), "qty": qty, "side": side, "entry": price})
        st.success("Position added.")
    if st.session_state.positions:
        df = pd.DataFrame(st.session_state.positions)
        st.dataframe(df)

# ---------- ALERTS TAB ----------
with tabs[3]:
    st.header("Price Alerts")
    a_sym = st.text_input("Symbol", key="alert_sym")
    a_price = st.number_input("Target price", value=0.0, key="alert_price")
    direction = st.selectbox("Direction", [">= (cross above)", "<= (cross below)"], key="alert_dir")
    if st.button("Create Alert"):
        st.session_state.alerts.append({
            "symbol": a_sym.upper(),
            "price": float(a_price),
            "dir": direction,
            "notified": False
        })
        st.success("Alert created.")
    if st.session_state.alerts:
        st.dataframe(pd.DataFrame(st.session_state.alerts))
    if st.button("Check Alerts"):
        for a in st.session_state.alerts:
            hist = get_history(a["symbol"], period="5d")
            cur = hist["Close"].iloc[-1] if not hist.empty else None
            if cur is not None:
                if a["dir"].startswith(">=") and cur >= a["price"] and not a["notified"]:
                    st.success(f"ALERT: {a['symbol']} {a['dir']} {a['price']} (Now {cur})")
                    a["notified"] = True
                if a["dir"].startswith("<=") and cur <= a["price"] and not a["notified"]:
                    st.success(f"ALERT: {a['symbol']} {a['dir']} {a['price']} (Now {cur})")
                    a["notified"] = True

# ---------- REPORTS TAB ----------
with tabs[4]:
    st.header("Reports & Export")
    if st.button("Generate Report"):
        df = pd.DataFrame(st.session_state.positions)
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        md = f"# Portfolio Report — {now}\n\n" + df.to_markdown(index=False)
        st.download_button("Download Markdown", md, file_name="portfolio.md")
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button("Download CSV", csv_buf.getvalue(), file_name="portfolio.csv")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("""
### ✅ Next Steps
- If you want even lower latency, I can add a streaming worker using Polygon/Finnhub WebSocket and save live ticks to Supabase/Redis. Streamlit will then read from the DB.
- Add RSS (Moneycontrol / ET / LiveMint) and Supabase persistence next.
""")
