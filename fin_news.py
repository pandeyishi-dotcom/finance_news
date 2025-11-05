# app.py
"""
Bloomberg-Lite — Feature-rich Streamlit Terminal (single-file scaffold)
Drop-in prototype: many features are implemented; paid/real-time sources are stubbed and documented.
Author: Generated for Ishani (you) — extend freely.
Run: streamlit run app.py
Streamlit secrets: add API keys to .streamlit/secrets.toml or Streamlit Cloud secrets.
"""

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import os
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Optional: sentiment / summarization using OpenAI (only if you add OPENAI_API_KEY to secrets)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Optional: ag-grid interactive table
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False

# --------------- Helper dataclasses -----------------
@dataclass
class Position:
    ticker: str
    qty: float
    entry_price: float
    side: str  # 'long'|'short'
    created_at: str

# --------------- App config / UI --------------------
st.set_page_config(page_title="Bloomberg-Lite", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    /* Dark theme */
    .reportview-container, .main, header, .stApp {
        background-color: #0B0C0D;
        color: #E6EEF3;
    }
    .css-1d391kg {background-color: #0B0C0D;}
    .stButton>button {background-color:#1f2937;color:#fff;}
    .stMetricDelta {font-weight:700;}
    .title {font-family: Inter, sans-serif;}
    a {color:#00BFFF;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Bloomberg-Lite — Terminal Prototype")

# --------------- Session state defaults ----------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AAPL", "MSFT", "TSLA", "RELIANCE.NS"]
if "positions" not in st.session_state:
    st.session_state.positions = []  # list[Position] serialized
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "command_history" not in st.session_state:
    st.session_state.command_history = []

# --------------- Utilities ----------------------------
def parse_command(cmd: str):
    """Simple command parser: COMMAND [ARGS]"""
    st.session_state.command_history.append({"cmd": cmd, "ts": datetime.utcnow().isoformat()})
    tokens = cmd.strip().split()
    if not tokens:
        return {"action": "noop"}
    verb = tokens[0].lower()
    args = tokens[1:]
    return {"action": verb, "args": args}

@st.cache_data(ttl=60)
def fetch_quote(ticker: str):
    """Return current quote and a small history (uses yfinance for prototype)"""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="1mo", interval="1d")
        # simplified quote
        quote = {
            "symbol": ticker,
            "last": info.get("currentPrice") or (hist["Close"].iloc[-1] if not hist.empty else None),
            "prev_close": info.get("previousClose"),
            "change": None,
            "pct": None,
            "info": info,
            "history": hist.reset_index()
        }
        try:
            if quote["prev_close"]:
                quote["change"] = quote["last"] - quote["prev_close"]
                quote["pct"] = (quote["change"] / quote["prev_close"]) * 100
        except Exception:
            pass
        return quote
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=300)
def fetch_multiple_quotes(tickers: List[str]) -> Dict[str, Any]:
    out = {}
    for t in tickers:
        out[t] = fetch_quote(t)
    return out

def humanize(x):
    if x is None:
        return "-"
    for unit in ["", "K", "M", "B", "T"]:
        if abs(x) < 1000:
            return f"{x:.2f}{unit}"
        x /= 1000.0
    return f"{x:.2f}Q"

# --------------- Sidebar: Controls --------------------
with st.sidebar:
    st.header("Command & Controls")
    cmd = st.text_input("Command (e.g., AAPL GO / TSLA 1Y CHART / ADD TICKER MSFT)", key="command_input")
    if st.button("Run Command"):
        result = parse_command(cmd)
        st.write("Parsed:", result)
        # handle a few built-in commands:
        action = result["action"]
        args = result["args"]
        if action in ["add", "addticker", "add_ticker", "add-ticker"]:
            if args:
                sym = args[0].upper()
                st.session_state.watchlist.append(sym)
                st.success(f"Added {sym} to watchlist")
        elif action in ["remove", "rm", "rmticker"]:
            if args:
                sym = args[0].upper()
                st.session_state.watchlist = [t for t in st.session_state.watchlist if t != sym]
                st.success(f"Removed {sym} from watchlist")
        elif action in ["go", "open"] and args:
            st.session_state["selected_ticker"] = args[0].upper()
        elif action in ["help", "?"]:
            st.write("Commands: ADD <TICKER>, REMOVE <TICKER>, <TICKER> GO, <TICKER> 1Y CHART")
        else:
            st.info("Command added to history.")
    st.markdown("---")

    st.subheader("Watchlist")
    wl_txt = st.text_area("Tickers (comma separated)", value=",".join(st.session_state.watchlist), key="wl_editor", height=120)
    if st.button("Save Watchlist"):
        new = [x.strip().upper() for x in wl_txt.split(",") if x.strip()]
        st.session_state.watchlist = new
        st.success("Watchlist updated.")
    st.markdown("---")
    st.subheader("Quick Actions")
    c1, c2 = st.columns(2)
    if c1.button("Add AAPL"): st.session_state.watchlist.append("AAPL")
    if c2.button("Add RELIANCE.NS"): st.session_state.watchlist.append("RELIANCE.NS")
    st.markdown("---")
    st.subheader("Settings")
    st.checkbox("Enable AI Summaries (experimental)", value=False, key="ai_enable")
    st.selectbox("Default chart timeframe", ["1d", "5d", "1mo", "3mo", "1y", "5y"], index=4, key="default_tf")
    st.markdown("API keys: set in Streamlit secrets as 'ALPHA_VANTAGE', 'OPENAI' if you plan to enable them.")
    st.markdown("---")
    st.write("Session saved locally. For multi-user add a backend DB (Supabase/Firebase).")

# --------------- Top market tape --------------------
def market_tape(quotes: Dict[str, Any]):
    items = []
    for s, data in quotes.items():
        if data.get("error"):
            continue
        last = data.get("last")
        pct = data.get("pct")
        arrow = "▲" if pct and pct > 0 else ("▼" if pct and pct < 0 else "")
        items.append(f"{s}: {last} {arrow} ({pct:+.2f}%)" if pct is not None else f"{s}: {last}")
    st.markdown("**Market Tape**")
    tape = "  —  ".join(items[:30])
    st.markdown(f"<div style='padding:8px;background:#0f1720;border-radius:6px'>{tape}</div>", unsafe_allow_html=True)

# --------------- Main layout ------------------------
left_col, center_col, right_col = st.columns([1.5, 3.5, 2])

# Left column: watchlist + screener
with left_col:
    st.subheader("Watchlist")
    quotes = fetch_multiple_quotes(st.session_state.watchlist)
    for sym, q in quotes.items():
        if q.get("error"):
            st.write(sym, "error:", q["error"])
            continue
        last = q.get("last")
        pct = q.get("pct")
        color = "green" if (pct or 0) >= 0 else "red"
        cols = st.columns([2,1])
        cols[0].markdown(f"**{sym}**")
        cols[1].metric("", f"{last}", delta=f"{pct:+.2f}%" if pct is not None else "")
    st.markdown("---")
    st.subheader("Screener (basic)")
    screener_pe = st.slider("Max PE", 5, 200, 100)
    # NOTE: basic screener using yfinance info (slow for many tickers)
    screener_results = []
    if st.button("Run Screener"):
        with st.spinner("Running screener (yfinance, may be slow)..."):
            for sym in st.session_state.watchlist:
                info = fetch_quote(sym).get("info", {})
                pe = info.get("trailingPE") or 9999
                if pe and pe <= screener_pe:
                    screener_results.append({"symbol": sym, "pe": pe, "marketCap": info.get("marketCap")})
        st.write(pd.DataFrame(screener_results))
    st.markdown("---")
    st.subheader("Mini Heatmap")
    # Heatmap using current pct from watchlist
    df_heat = []
    for sym, q in quotes.items():
        pct = q.get("pct") or 0
        df_heat.append({"symbol": sym, "pct": pct})
    if df_heat:
        heat = pd.DataFrame(df_heat)
        fig = px.treemap(heat, path=["symbol"], values=[1]*len(heat), color="pct", color_continuous_scale=px.colors.diverging.RdYlGn[::-1])
        st.plotly_chart(fig, use_container_width=True)

# Center column: chart + indicators + portfolio simulator
with center_col:
    st.subheader("Main")
    selected = st.selectbox("Select ticker", options=st.session_state.watchlist, key="main_select", index=0)
    tf = st.selectbox("Timeframe", ["1d","5d","1mo","3mo","1y","5y"], index=4, key="main_tf")
    interval_map = {"1d":"1m","5d":"5m","1mo":"30m","3mo":"1d","1y":"1d","5y":"1d"}
    interval = interval_map.get(tf, "1d")

    quote = fetch_quote(selected)
    if quote.get("error"):
        st.error("Error fetching quote: " + quote["error"])
    else:
        last = quote.get("last")
        prev = quote.get("prev_close") or last
        delta = (last - prev) if (prev and last) else 0
        st.metric(f"{selected} price", value=f"{last}", delta=f"{(delta/prev*100):+.2f}%" if prev else "n/a")

        # Chart area
        hist = quote.get("history")
        if hist is None or hist.empty:
            st.info("No history available for this ticker.")
        else:
            # Plotly candlestick + volume + moving averages
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], mode='lines', name='Close', line=dict(width=1)))
            # 20 & 50 day moving averages (if available)
            hist["MA20"] = hist["Close"].rolling(20).mean()
            hist["MA50"] = hist["Close"].rolling(50).mean()
            fig.add_trace(go.Scatter(x=hist['Date'], y=hist['MA20'], mode='lines', name='MA20', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=hist['Date'], y=hist['MA50'], mode='lines', name='MA50', line=dict(dash='dot')))
            fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Indicators selector
            with st.expander("Indicators"):
                sma = st.checkbox("Show SMA 200", value=False)
                # TODO: implement more indicators and overlays
                if sma:
                    hist["SMA200"] = hist["Close"].rolling(200).mean()
                    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['SMA200'], mode='lines', name='SMA200'))

    st.markdown("---")
    # Portfolio / Paper trading
    st.subheader("Paper Trading Simulator")
    cols = st.columns([1,1,1,1])
    qty = cols[0].number_input("Qty", min_value=1, value=1, step=1)
    side = cols[1].selectbox("Side", ["long","short"])
    price_input = cols[2].number_input("Price (leave 0 for market)", min_value=0.0, value=0.0, step=0.01)
    if cols[3].button("Place Trade"):
        entry = price_input if price_input > 0 else (quote.get("last") or 0)
        pos = Position(ticker=selected, qty=qty, entry_price=entry, side=side, created_at=datetime.utcnow().isoformat())
        st.session_state.positions.append(asdict(pos))
        st.success(f"Paper trade placed: {side} {qty} {selected} @ {entry:.2f}")
    # Show positions
    if st.session_state.positions:
        st.write("Open Positions")
        pos_df = pd.DataFrame(st.session_state.positions)
        st.table(pos_df)

# Right column: news, fundamentals, AI assistant
with right_col:
    st.subheader("News (prototype)")
    # yfinance news (limited). For production, use NewsAPI / Bloomberg / vendor.
    try:
        news_list = yf.Ticker(selected).news or []
    except Exception:
        news_list = []
    if not news_list:
        st.info("No news from yfinance; add news provider keys to use NewsAPI / custom scrapers.")
    else:
        for n in news_list[:10]:
            st.markdown(f"**{n.get('title','-')}**")
            st.write(n.get('publisher',''), n.get('link',''))
    st.markdown("---")
    st.subheader("Fundamentals (quick)")
    info = quote.get("info", {})
    fundamentals = {
        "Market Cap": humanize(info.get("marketCap")) if info else "-",
        "Trailing P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "Dividend Yield": info.get("dividendYield"),
        "52W High / Low": f"{info.get('fiftyTwoWeekHigh')} / {info.get('fiftyTwoWeekLow')}"
    }
    st.json(fundamentals)

    st.markdown("---")
    st.subheader("AI Copilot (optional)")
    if st.session_state.ai_enable and OPENAI_AVAILABLE and st.secrets.get("OPENAI_API_KEY"):
        st.info("AI is enabled. Summarizing latest news...")
        try:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            combined = "\n".join([n.get("title","") + " -- " + (n.get("summary", "") or "") for n in news_list[:5]])
            prompt = f"Summarize these headlines into 3 concise bullets, focus on market impact:\n\n{combined}"
            resp = client.responses.create(model="gpt-4o-mini", input=prompt)
            summary_text = resp.output_text if hasattr(resp, "output_text") else str(resp)
            st.markdown(summary_text)
        except Exception as e:
            st.error("AI summarization failed: " + str(e))
    else:
        st.caption("Enable OpenAI in Streamlit secrets as OPENAI_API_KEY and check 'Enable AI Summaries' to use the Copilot.")

# --------------- Alerts (background-ish) ----------------
st.markdown("---")
st.header("Alerts & Notifications")
with st.expander("Manage alerts"):
    st.write("Create price alerts (in-app demo). For real push alerts use email/telegram integrations.")
    new_alert_ticker = st.text_input("Ticker for alert", value="")
    new_alert_price = st.number_input("Target price", value=0.0)
    if st.button("Create Alert"):
        if new_alert_ticker and new_alert_price > 0:
            st.session_state.alerts.append({"ticker": new_alert_ticker.upper(), "price": new_alert_price, "created": datetime.utcnow().isoformat(), "notified": False})
            st.success("Alert created.")
    # Show alerts
    if st.session_state.alerts:
        for a in st.session_state.alerts:
            st.write(a)
            try:
                q = fetch_quote(a["ticker"])
                last = q.get("last")
                if last and not a.get("notified") and ((last >= a["price"] and True) or False):
                    st.info(f"ALERT: {a['ticker']} crossed {a['price']} — last {last}")
                    a["notified"] = True
            except Exception:
                pass

# --------------- Data Explorer / Raw tables ----------------
st.markdown("---")
st.header("Data Explorer")
with st.expander("Raw time series & download"):
    if quote.get("history") is not None:
        st.write("Download CSV of history")
        csv = quote["history"].to_csv(index=False).encode("utf-8")
        st.download_button(label="Download CSV", data=csv, file_name=f"{selected}_history.csv", mime="text/csv")
        st.dataframe(quote["history"].tail(20))
    else:
        st.write("No timeseries to show.")

# --------------- Plugins loader (basic) ----------------
st.markdown("---")
st.subheader("Plugin Manager (dev)")
plugins_dir = "plugins"
if not os.path.exists(plugins_dir):
    os.makedirs(plugins_dir)
st.write("Drop small python plugin files into `/plugins` and they will appear here (simple sandbox).")
plugin_files = [f for f in os.listdir(plugins_dir) if f.endswith(".py")]
for pf in plugin_files:
    st.write("Plugin:", pf)
    if st.button(f"Run {pf}"):
        try:
            # VERY simple and naive plugin loader: exec in restricted namespace
            ns = {"st": st, "fetch_quote": fetch_quote, "quotes": quotes}
            with open(os.path.join(plugins_dir, pf), "r") as f:
                code = f.read()
            exec(code, ns)  # plugins can call st.* and access fetch_quote
        except Exception as e:
            st.error(f"Plugin error: {e}")

# --------------- Footer: shortcuts & tips ----------------
st.markdown("---")
st.caption("Shortcuts: type commands in Command box. Try `ADD TSLA`, `AAPL GO`, `RELIANCE.NS GO`.")
st.caption("For production: replace yfinance data with Polygon / IEX, add Redis caching, and secure API keys in Streamlit secrets.")
