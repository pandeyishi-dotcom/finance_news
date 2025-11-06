# bloomberg_lite_advanced.py
# Bloomberg-Lite vX — Advanced single-file prototype
# Features: simulated live ticks, heatmap, news+sentiment, AI assistant (optional),
# technical indicators, portfolio analytics, backtest, simple forecasting, alerts, reports.
#
# Safe: optional deps guarded. Runs out-of-the-box with the requirements listed below.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import json
import io
from datetime import datetime, timedelta
import plotly.express as px
import threading
import os
import math

# Optional OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ------------- Basic page setup -------------
st.set_page_config("Bloomberg-Lite vX", layout="wide", initial_sidebar_state="expanded")
st.title("📡 Bloomberg-Lite vX — Advanced Market Intelligence Prototype")
st.markdown("A modular prototype: live ticks (simulated), heatmap, news+sentiment, AI assistant (optional), indicators, portfolio analytics, backtest, forecast, alerts, and reports.")

# ------------- defaults & session storage -------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AAPL", "MSFT", "GOOG", "TSLA", "RELIANCE.NS", "INFY.NS"]
if "ticks" not in st.session_state:
    # latest tick for each symbol
    st.session_state.ticks = {s: {"price": None, "ts": None} for s in st.session_state.watchlist}
if "sim_base" not in st.session_state:
    # internal base prices for simulation random walk
    st.session_state.sim_base = {s: float(100 + np.random.rand()*200) for s in st.session_state.watchlist}
if "positions" not in st.session_state:
    st.session_state.positions = []  # list of dicts
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "history_cache" not in st.session_state:
    st.session_state.history_cache = {}
if "live_running" not in st.session_state:
    st.session_state.live_running = False

# optional local persistence file (for local use)
DATA_STORE = "data_store.json"
def save_local_store():
    try:
        obj = {
            "watchlist": st.session_state.watchlist,
            "positions": st.session_state.positions,
            "alerts": st.session_state.alerts
        }
        with open(DATA_STORE, "w") as f:
            json.dump(obj, f)
    except Exception:
        pass

def load_local_store():
    if os.path.exists(DATA_STORE):
        try:
            with open(DATA_STORE, "r") as f:
                obj = json.load(f)
                st.session_state.watchlist = obj.get("watchlist", st.session_state.watchlist)
                st.session_state.positions = obj.get("positions", st.session_state.positions)
                st.session_state.alerts = obj.get("alerts", st.session_state.alerts)
        except Exception:
            pass

# Attempt to load (safe)
load_local_store()

# ------------- Utilities & simplified indicators -------------
def safe_get_history(symbol, period="1y", interval="1d"):
    key = f"{symbol}_{period}_{interval}"
    if key in st.session_state.history_cache:
        return st.session_state.history_cache[key].copy()
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df is None or df.empty:
            df = pd.DataFrame()
        else:
            df = df.reset_index()
        st.session_state.history_cache[key] = df
        return df.copy()
    except Exception:
        return pd.DataFrame()

def compute_sma(df, window):
    return df["Close"].rolling(window).mean()

def compute_ema(df, span):
    return df["Close"].ewm(span=span, adjust=False).mean()

def compute_rsi(df, period=14):
    # classic RSI implementation
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

# Simple sentiment lexicon (tiny)
POS_WORDS = {"gain","up","beat","beats","beats expectations","positive","improve","growth","surge","soar","upgrade"}
NEG_WORDS = {"loss","down","miss","missed","cuts","downgrade","decline","fall","drop","weak"}

def simple_sentiment_from_headlines(headlines):
    # headlines: list of strings
    score = 0.0
    for t in headlines:
        txt = t.lower()
        for w in POS_WORDS:
            if w in txt:
                score += 1
        for w in NEG_WORDS:
            if w in txt:
                score -= 1
    # normalize
    if len(headlines) > 0:
        return score / len(headlines)
    return 0.0

# ------------- Live simulation worker (threaded) -------------
def sim_tick_step():
    # update st.session_state.ticks with a small random walk
    for s in st.session_state.watchlist:
        base = st.session_state.sim_base.get(s, 100.0)
        # random walk: small Gaussian movement scaled by base
        change_pct = np.random.normal(loc=0.0, scale=0.0015)  # ~0.15% std
        new_price = max(0.01, base * (1 + change_pct))
        st.session_state.sim_base[s] = new_price
        st.session_state.ticks[s] = {"price": round(new_price, 2), "ts": datetime.utcnow().isoformat()}
    # keep worker quick
    # do NOT sleep here; main loop controls timing

# Thread runner (non-blocking)
def start_simulator(interval=1.0):
    if st.session_state.live_running:
        return
    st.session_state.live_running = True
    def run():
        while st.session_state.live_running:
            sim_tick_step()
            time.sleep(interval)
    t = threading.Thread(target=run, daemon=True)
    t.start()

def stop_simulator():
    st.session_state.live_running = False

# ------------- UI sidebar: global controls -------------
with st.sidebar:
    st.header("Controls")
    st.write("Simulation / Data & Keys")
    run_live = st.checkbox("Run live simulator", value=False)
    if run_live:
        start_simulator(interval=1.0)
    else:
        stop_simulator()
    st.write("---")
    st.markdown("**Watchlist (comma separated)**")
    wl_old = ",".join(st.session_state.watchlist)
    wl_new = st.text_area("Edit watchlist", value=wl_old, height=80)
    if st.button("Save watchlist"):
        new = [x.strip().upper() for x in wl_new.split(",") if x.strip()]
        if new:
            st.session_state.watchlist = new
            # re-init ticks/bases
            for s in new:
                if s not in st.session_state.sim_base:
                    st.session_state.sim_base[s] = float(100 + np.random.rand()*200)
                if s not in st.session_state.ticks:
                    st.session_state.ticks[s] = {"price": None, "ts": None}
            st.success("Watchlist saved.")
            save_local_store()
    st.write("---")
    st.markdown("**API Keys (optional)**")
    st.caption("Add OPENAI_API_KEY to enable the AI assistant.")
    # note: in Streamlit Cloud, use Settings -> Secrets instead of typing here.
    openai_key = st.text_input("OpenAI key (local test only)", type="password")
    if openai_key:
        st.session_state.temp_openai = openai_key
    # persistence
    if st.button("Save session locally (file)"):
        save_local_store()
        st.success("Saved locally (data_store.json)")

# ------------- Top live tape -------------
st.subheader("Live Market Tape (simulated)")
cols = st.columns([1,4,1])
with cols[1]:
    tape_text = []
    for s in st.session_state.watchlist:
        tick = st.session_state.ticks.get(s, {})
        price = tick.get("price")
        ts = tick.get("ts")
        if price is None:
            # try initial fetch from yfinance
            try:
                hist = safe_get_history(s, period="5d", interval="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
                    st.session_state.ticks[s] = {"price": round(price,2), "ts": datetime.utcnow().isoformat()}
                else:
                    price = "-"
            except Exception:
                price = "-"
        tape_text.append(f"{s} {price}")
    st.markdown("  •  ".join(tape_text))

# ------------- Tabs: Multi-feature layout -------------
tabs = st.tabs(["Overview","Heatmap","News+Sentiment","AI Assistant","Technicals","Portfolio","Backtest","Forecast","Alerts","Reports"])

# ---------- TAB: Overview ----------
with tabs[0]:
    st.header("Overview — Quick snapshots")
    # top gainers/losers from simulated ticks vs previous day's close (yfinance)
    rows = []
    for s in st.session_state.watchlist:
        price = st.session_state.ticks.get(s,{}).get("price")
        # get prev close from yfinance history quick
        try:
            hist = safe_get_history(s, period="5d", interval="1d")
            prev_close = None
            if not hist.empty:
                prev_close = float(hist["Close"].iloc[-1])
        except Exception:
            prev_close = None
        change = None
        if price is not None and prev_close is not None:
            change = (price - prev_close) / prev_close * 100
        rows.append({"symbol": s, "price": price, "prev_close": prev_close, "pct": change})
    df_ov = pd.DataFrame(rows)
    if not df_ov.empty:
        st.dataframe(df_ov.sort_values(by="pct", ascending=False).fillna("-"))

# ---------- TAB: Heatmap ----------
with tabs[1]:
    st.header("Heatmap / Treemap of Watchlist")
    # compute % change for treemap
    treemap_rows = []
    for s in st.session_state.watchlist:
        price = st.session_state.ticks.get(s,{}).get("price")
        hist = safe_get_history(s, period="5d", interval="1d")
        prev = None
        if not hist.empty:
            prev = float(hist["Close"].iloc[-1])
        pct = None
        if price is not None and prev is not None and prev!=0:
            pct = (price - prev)/prev*100
        marketcap = None
        info = {}
        try:
            info = yf.Ticker(s).info or {}
            marketcap = info.get("marketCap") or (info.get("market_cap") if info.get("market_cap") else None)
        except Exception:
            marketcap = None
        treemap_rows.append({"symbol": s, "pct": pct or 0.0, "marketcap": marketcap or 1})
    treedf = pd.DataFrame(treemap_rows)
    if treedf.empty:
        st.info("No data for heatmap.")
    else:
        fig = px.treemap(treedf, path=["symbol"], values="marketcap", color="pct", color_continuous_scale="RdYlGn", title="Watchlist Heatmap: color by % change")
        st.plotly_chart(fig, use_container_width=True)

# ---------- TAB: News + Sentiment ----------
with tabs[2]:
    st.header("News & Lightweight Sentiment")
    news_sym = st.selectbox("Choose ticker for news", st.session_state.watchlist, index=0, key="news_sym")
    headlines = []
    try:
        raw = yf.Ticker(news_sym).news or []
        for n in raw[:15]:
            title = n.get("title") if isinstance(n, dict) else str(n)
            provider = n.get("publisher") if isinstance(n, dict) else ""
            link = n.get("link") if isinstance(n, dict) else ""
            headlines.append({"title": title, "provider": provider, "link": link})
    except Exception:
        headlines = []
    if headlines:
        cols = st.columns([3,1])
        with cols[0]:
            for h in headlines:
                st.markdown(f"**{h['title']}**")
                if h.get("provider"):
                    st.caption(h["provider"])
                if h.get("link"):
                    st.write(h["link"])
                st.markdown("---")
        with cols[1]:
            # compute sentiment
            hs = [h["title"] for h in headlines]
            sscore = simple_sentiment_from_headlines(hs)
            st.metric("Headline Sentiment (simple)", f"{sscore:+.2f}")
            st.markdown("**Top headline words (simple)**")
            words = " ".join(hs).lower().split()
            wc = pd.Series(words).value_counts().head(20)
            st.table(wc.rename_axis("word").reset_index(name="count").head(10))
    else:
        st.info("No headlines found via yfinance. Hook NewsAPI for richer feeds.")

# ---------- TAB: AI Assistant ----------
with tabs[3]:
    st.header("AI Research Assistant (optional)")
    st.markdown("Ask short market research questions. Requires OpenAI key (put in Streamlit secrets or enter in sidebar for local testing).")
    user_q = st.text_input("Ask the assistant", value="", key="ai_q")
    if st.button("Ask"):
        if OPENAI_AVAILABLE and ("OPENAI_API_KEY" in st.secrets or st.session_state.get("temp_openai")):
            key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("temp_openai")
            try:
                openai.api_key = key
                prompt = f"You are a market analyst. Answer concisely and show steps. Question: {user_q}"
                resp = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=300, temperature=0.2)
                answer = resp.choices[0].text.strip()
                st.markdown("**Assistant answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"OpenAI call failed: {e}")
        else:
            st.info("OpenAI not configured. Fallback: simple data-driven answer (limited).")
            # very basic fallback: try to interpret simple queries
            q = user_q.lower()
            if "top" in q and "gain" in q:
                # show top pct in watchlist
                rows = []
                for s in st.session_state.watchlist:
                    p = st.session_state.ticks.get(s,{}).get("price")
                    hist = safe_get_history(s, period="5d", interval="1d")
                    prev = float(hist["Close"].iloc[-1]) if not hist.empty else None
                    if p and prev:
                        rows.append((s, (p-prev)/prev*100))
                if rows:
                    rows = sorted(rows, key=lambda x: -x[1])
                    st.write(f"Top mover: {rows[0][0]} {rows[0][1]:+.2f}%")
                else:
                    st.write("No sufficient data for fallback answer.")
            else:
                st.write("Fallback can't parse this question. Provide OpenAI key for advanced answers.")

# ---------- TAB: Technical indicators ----------
with tabs[4]:
    st.header("Technicals — SMA/EMA/RSI/MACD")
    tech_sym = st.selectbox("Ticker", st.session_state.watchlist, key="tech_sym")
    hist = safe_get_history(tech_sym, period="2y", interval="1d")
    if hist.empty:
        st.info("No history available for this ticker.")
    else:
        short = st.number_input("Short window (SMA)", 10, 200, 20, key="sma_short")
        long = st.number_input("Long window (SMA)", 20, 400, 50, key="sma_long")
        hist["SMA_short"] = compute_sma(hist, short)
        hist["SMA_long"] = compute_sma(hist, long)
        hist["EMA_short"] = compute_ema(hist, short)
        hist["EMA_long"] = compute_ema(hist, long)
        hist["RSI"] = compute_rsi(hist, period=14)
        macd, macd_sig, macd_hist = compute_macd(hist)
        hist["MACD"] = macd
        hist["MACD_sig"] = macd_sig
        fig = px.line(hist, x="Date", y=["Close","SMA_short","SMA_long"], title=f"{tech_sym} Price & SMAs")
        st.plotly_chart(fig, use_container_width=True)
        st.line_chart(hist.set_index("Date")[["RSI"]].tail(200))
        st.line_chart(hist.set_index("Date")[["MACD","MACD_sig"]].tail(200))

# ---------- TAB: Portfolio ----------
with tabs[5]:
    st.header("Portfolio & Analytics")
    c1,c2,c3 = st.columns(3)
    with c1:
        psym = st.text_input("Add position symbol")
    with c2:
        pqty = st.number_input("Qty", min_value=1, value=1)
    with c3:
        pside = st.selectbox("Side", ["long","short"])
    if st.button("Add position"):
        price = st.session_state.ticks.get(psym,{}).get("price") or (safe_get_history(psym, period="5d")["Close"].iloc[-1] if not safe_get_history(psym, period="5d").empty else None) or 0.0
        st.session_state.positions.append({"symbol": psym.upper(), "qty": int(pqty), "side": pside, "entry": float(price), "added": datetime.utcnow().isoformat()})
        save_local_store()
        st.success("Position added.")
    if st.session_state.positions:
        dfpos = pd.DataFrame(st.session_state.positions)
        st.write("Positions")
        st.dataframe(dfpos)
        # compute P&L
        rows = []
        total = 0.0
        for p in st.session_state.positions:
            cur = st.session_state.ticks.get(p["symbol"],{}).get("price") or (safe_get_history(p["symbol"],period="5d")["Close"].iloc[-1] if not safe_get_history(p["symbol"],period="5d").empty else 0.0)
            pnl = (cur - p["entry"]) * p["qty"] * (1 if p["side"]=="long" else -1)
            rows.append({"symbol":p["symbol"], "qty":p["qty"], "entry":p["entry"], "current":cur, "pnl":pnl})
            total += pnl
        st.metric("Total Unrealized P&L", f"₹{total:,.2f}")
        st.dataframe(pd.DataFrame(rows))
        # correlation matrix of returns (if we have history)
        hist_map = {}
        for s in set([p["symbol"] for p in st.session_state.positions]):
            h = safe_get_history(s, period="6mo")
            if not h.empty:
                h = h.set_index("Date")["Close"].pct_change().dropna()
                hist_map[s] = h
        if hist_map:
            dfret = pd.DataFrame(hist_map)
            corr = dfret.corr()
            st.write("Correlation matrix")
            st.dataframe(corr)
            fig = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

# ---------- TAB: Backtest ----------
with tabs[6]:
    st.header("Backtest Lab — MA Crossover")
    bt_sym = st.selectbox("Backtest symbol", st.session_state.watchlist, index=0, key="bt_sym")
    short_ma = st.number_input("Short MA", 5, 100, 20, key="bt_short")
    long_ma = st.number_input("Long MA", 10, 300, 50, key="bt_long")
    capital = st.number_input("Initial capital", value=100000, step=1000, key="bt_cap")
    if st.button("Run Backtest"):
        df = safe_get_history(bt_sym, period="2y")
        if df.empty:
            st.error("No data for backtest.")
        else:
            df = df.copy().dropna(subset=["Close"])
            df["short"] = df["Close"].rolling(short_ma).mean()
            df["long"] = df["Close"].rolling(long_ma).mean()
            df["signal"] = 0
            df.loc[df["short"] > df["long"], "signal"] = 1
            df["positions"] = df["signal"].diff().fillna(0)
            cash = capital
            shares = 0
            nav = []
            for idx, row in df.iterrows():
                price = row["Close"]
                if row["positions"] == 1 and cash > price:
                    shares = cash // price
                    cash -= shares * price
                elif row["positions"] == -1 and shares > 0:
                    cash += shares * price
                    shares = 0
                nav.append(cash + shares * price)
            df["nav"] = nav
            st.line_chart(df.set_index("Date")["nav"])
            st.success("Backtest complete (demo). For production-grade backtests use vectorbt/backtrader.")

# ---------- TAB: Forecast ----------
with tabs[7]:
    st.header("Simple Forecast (moving-average projection)")
    fc_sym = st.selectbox("Forecast symbol", st.session_state.watchlist, key="fc_sym")
    window = st.number_input("Projection MA window (days)", 3, 60, 14)
    horizon = st.number_input("Horizon (days)", 1, 60, 7)
    if st.button("Run Forecast"):
        df = safe_get_history(fc_sym, period="180d")
        if df.empty:
            st.error("No data")
        else:
            last_ma = df["Close"].rolling(window).mean().iloc[-1]
            # simple projection: use MA as base and assume small noise
            future_dates = [df["Date"].iloc[-1] + pd.Timedelta(days=i) for i in range(1, horizon+1)]
            proj = []
            for i in range(horizon):
                noise = np.random.normal(0, 0.01)
                last_ma = last_ma * (1 + noise)
                proj.append(last_ma)
            out = pd.DataFrame({"date": future_dates, "proj_close": proj})
            st.line_chart(pd.concat([df.set_index("Date")["Close"].tail(60), pd.Series(proj, index=future_dates)], axis=0))

# ---------- TAB: Alerts ----------
with tabs[8]:
    st.header("Alerts")
    a_sym = st.text_input("Symbol for alert", key="alert_sym_input")
    a_price = st.number_input("Target price", value=0.0, key="alert_price_input")
    a_dir = st.selectbox("Trigger direction", [">= (cross above)", "<= (cross below)"], key="alert_dir_input")
    if st.button("Create Alert"):
        st.session_state.alerts.append({"symbol": a_sym.upper(), "price": float(a_price), "dir": a_dir, "notified": False, "created": datetime.utcnow().isoformat()})
        save_local_store()
        st.success("Alert created.")
    if st.session_state.alerts:
        st.dataframe(pd.DataFrame(st.session_state.alerts))
    if st.button("Check Alerts Now"):
        triggered = []
        for a in st.session_state.alerts:
            cur = st.session_state.ticks.get(a["symbol"],{}).get("price")
            if cur is None:
                try:
                    cur = float(safe_get_history(a["symbol"], period="5d")["Close"].iloc[-1])
                except Exception:
                    cur = None
            if cur is None:
                continue
            if a["dir"].startswith(">=") and cur >= a["price"] and not a.get("notified"):
                triggered.append((a, cur))
                a["notified"] = True
            if a["dir"].startswith("<=") and cur <= a["price"] and not a.get("notified"):
                triggered.append((a, cur))
                a["notified"] = True
        if triggered:
            for a,cur in triggered:
                st.success(f"ALERT: {a['symbol']} {a['dir']} {a['price']} — now {cur}")
        else:
            st.info("No alerts triggered.")

# ---------- TAB: Reports ----------
with tabs[9]:
    st.header("Reports & Export")
    if st.button("Export positions to CSV & Markdown"):
        df = pd.DataFrame(st.session_state.positions)
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        md = f"# Portfolio Snapshot — {datetime.utcnow().isoformat()}\n\n" + df.to_markdown(index=False)
        st.download_button("Download CSV", csv_buf.getvalue(), file_name=f"positions_{datetime.utcnow().date()}.csv")
        st.download_button("Download MD", md, file_name=f"positions_{datetime.utcnow().date()}.md")
        st.success("Exports ready.")

# ---------- Footer & TODOs ----------
st.markdown("---")
st.markdown("""
**Notes & next steps (developer):**
- This prototype simulates live ticks. Replace simulator with Polygon WebSocket worker & Redis for real-time production.
- Swap yfinance history calls with a cached fundamentals DB for fast screening.
- For pro backtesting, integrate `vectorbt` or `backtrader`.
- Add Supabase or Firebase for cloud persistence and multi-user features.
- Add third-party sentiment (X/Twitter + NewsAPI) and embeddings (OpenAI embeddings) for robust signal generation.
""")

# End of file
