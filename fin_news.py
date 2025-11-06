# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io
import time
from datetime import datetime
import feedparser
import re
from collections import Counter

# Optional auto refresh helper
try:
    from streamlit_autorefresh import st_autorefresh
    AUTORELOAD_AVAILABLE = True
except Exception:
    AUTORELOAD_AVAILABLE = False

st.set_page_config(page_title="Bloomberg-Lite (API-free)", layout="wide", initial_sidebar_state="expanded")
st.title("📡 Bloomberg-Lite — API-free News & Market Dashboard")

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
    st.session_state.live_prices = {}

# ---------- UTILITIES ----------
@st.cache_data(ttl=300)
def get_history(sym, period="1y", interval="1d"):
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
    sym = (sym or "").strip()
    if not sym:
        return None
    try:
        df = yf.Ticker(sym).history(period="1d", interval="1m")
        if df is None or df.empty or "Close" not in df.columns:
            df2 = yf.Ticker(sym).history(period="5d", interval="1d")
            if df2 is None or df2.empty or "Close" not in df2.columns:
                return None
            return float(df2["Close"].iloc[-1])
        return float(df["Close"].iloc[-1])
    except Exception:
        return None

# ---------- News (RSS) ----------
@st.cache_data(ttl=300)
def fetch_google_news_rss(q="finance OR markets OR stocks", max_items=12):
    """
    Fetch headlines from Google News RSS. Returns list of dicts with title, link, published, summary.
    """
    try:
        query = q.replace(" ", "+")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        results = []
        for e in feed.entries[:max_items]:
            results.append({
                "title": getattr(e, "title", "") or "",
                "link": getattr(e, "link", "") or "",
                "published": getattr(e, "published", "") or "",
                "summary": getattr(e, "summary", "") or ""
            })
        return results
    except Exception:
        return []

# ---------- Local summarizer (no external API) ----------
STOPWORDS = {
    "the","a","an","and","or","for","to","in","on","of","is","are","as","by","with","from",
    "after","before","at","that","this","it","its","new","will","be","have","has","was","were"
}

RISE_WORDS = {"rise","rises","rose","jump","jumps","surge","gains","gain","up","soar","record"}
FALL_WORDS = {"fall","falls","fell","drop","drops","decline","declines","slump","down","slide","tumble","loss","losses"}

def tokenize(text):
    text = re.sub(r"https?://\S+", "", text)  # remove urls
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    tokens = [t.lower() for t in text.split() if t.strip()]
    return tokens

def top_keywords(headlines, n=6):
    cnt = Counter()
    for h in headlines:
        for t in tokenize(h):
            if len(t) > 2 and t not in STOPWORDS and not t.isnumeric():
                cnt[t] += 1
    return [k for k, _ in cnt.most_common(n)]

def detect_tone(headlines):
    """Simple heuristic to detect market tone across headlines."""
    score = 0
    for h in headlines:
        tokens = set(tokenize(h))
        if tokens & RISE_WORDS:
            score += 1
        if tokens & FALL_WORDS:
            score -= 1
    if score > 0:
        return "Bullish"
    if score < 0:
        return "Bearish"
    return "Neutral"

def local_summary_from_headlines(headlines):
    """
    Create 3 concise bullets:
    1) Theme from top keywords
    2) Representative top headline
    3) Tone: Bullish/Neutral/Bearish
    """
    if not headlines:
        return "No headlines available to summarize."
    titles = [h for h in headlines if h]
    kw = top_keywords(titles, n=6)
    theme = " ".join(kw[:4]) if kw else ""
    rep = titles[0] if titles else ""
    tone = detect_tone(titles)
    bullets = []
    if theme:
        bullets.append(f"• Main themes: {theme}.")
    if rep:
        short = rep if len(rep) <= 180 else rep[:177] + "…"
        bullets.append(f"• Example headline: {short}")
    bullets.append(f"• Market tone (heuristic): **{tone}**.")
    return "\n".join(bullets)

# ---------- LAYOUT ----------
tabs = st.tabs(["Market", "News Hub", "Portfolio", "Alerts", "Reports"])

# ---------- MARKET TAB ----------
with tabs[0]:
    st.header("Market Overview (API-free)")
    wl = st.text_area("Watchlist (comma separated)", ",".join(st.session_state.watchlist), height=80)
    if st.button("Update Watchlist"):
        st.session_state.watchlist = [s.strip().upper() for s in wl.split(",") if s.strip()]
        st.success("Watchlist updated.")

    sel = st.selectbox("Select ticker", st.session_state.watchlist)
    col1, col2, col3 = st.columns([3,1,1])
    with col2:
        interval = st.number_input("Poll interval (secs)", min_value=5, value=15, step=1)
    with col3:
        if st.session_state.polling:
            if st.button("Stop Live Polling"):
                st.session_state.polling = False
                st.success("Polling stopped.")
        else:
            if st.button("Start Live Polling"):
                st.session_state.polling = True
                st.success("Polling started.")

    if st.session_state.polling and AUTORELOAD_AVAILABLE:
        st_autorefresh(interval=int(interval*1000), key="autorefresh-market")

    if st.session_state.polling:
        for s in st.session_state.watchlist:
            price = get_latest_price(s)
            if price is not None:
                st.session_state.live_prices[s] = {"price": price, "ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}

    if st.session_state.live_prices:
        rows = []
        for s in st.session_state.watchlist:
            info = st.session_state.live_prices.get(s)
            rows.append({"symbol": s, "last_price": info["price"] if info else None, "updated_at": info["ts"] if info else None})
        st.dataframe(pd.DataFrame(rows))
    else:
        st.info("No live prices yet. Start polling or press 'Refresh prices now'.")

    if st.button("Refresh prices now"):
        p = get_latest_price(sel)
        if p is not None:
            st.session_state.live_prices[sel] = {"price": p, "ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
            st.success(f"{sel} — {p}")
        else:
            st.error(f"Could not fetch latest price for {sel}.")

    hist = get_history(sel)
    if not hist.empty and "Close" in hist.columns:
        if "Date" in hist.columns:
            st.line_chart(hist.set_index("Date")["Close"])
        else:
            st.line_chart(hist["Close"])
    else:
        st.info("No historical data to show for selected ticker.")

# ---------- NEWS HUB ----------
with tabs[1]:
    st.header("📰 Finance & Investing News (RSS, no APIs)")
    q = st.text_input("Search query for news RSS", value="finance OR markets OR stocks")
    if st.button("Fetch News"):
        with st.spinner("Fetching RSS headlines..."):
            news_list = fetch_google_news_rss(q=q, max_items=12)
            st.session_state.news_list = news_list
    else:
        news_list = st.session_state.get("news_list", fetch_google_news_rss(q=q, max_items=12))

    if news_list:
        for n in news_list:
            st.markdown(f"**{n['title']}**  \n*{n['published']}*  \n[{n['link']}]({n['link']})")
            if n.get("summary"):
                st.markdown(f"_{n['summary']}_")
            st.markdown("---")
    else:
        st.info("No news found. Try a different query or click Fetch News.")

    st.markdown("### 🧠 Local Summary (no external AI)")
    if st.button("Generate Local Summary"):
        titles = [n["title"] for n in news_list] if news_list else []
        summary = local_summary_from_headlines(titles)
        st.subheader("Local summary")
        st.write(summary)

# ---------- PORTFOLIO ----------
with tabs[2]:
    st.header("Portfolio Simulator")
    sym = st.text_input("Symbol to add")
    qty = st.number_input("Qty", min_value=1, value=1)
    side = st.selectbox("Side", ["long", "short"])
    if st.button("Add Position"):
        hist = get_history(sym, period="5d")
        price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        st.session_state.positions.append({"symbol": (sym or "").upper(), "qty": int(qty), "side": side, "entry": float(price)})
        st.success("Position added.")
    if st.session_state.positions:
        st.dataframe(pd.DataFrame(st.session_state.positions))
    else:
        st.info("No positions yet. Add one above.")

# ---------- ALERTS ----------
with tabs[3]:
    st.header("Price Alerts")
    a_sym = st.text_input("Symbol", key="alert_sym")
    a_price = st.number_input("Target price", value=0.0, key="alert_price")
    direction = st.selectbox("Direction", [">= (cross above)", "<= (cross below)"], key="alert_dir")
    if st.button("Create Alert"):
        if not a_sym.strip():
            st.error("Enter symbol to create alert.")
        else:
            st.session_state.alerts.append({"symbol": a_sym.strip().upper(), "price": float(a_price), "dir": direction, "notified": False})
            st.success("Alert created.")
    if st.session_state.alerts:
        st.dataframe(pd.DataFrame(st.session_state.alerts))
    else:
        st.info("No alerts configured.")

    if st.button("Check Alerts"):
        any_alert = False
        for a in st.session_state.alerts:
            sym = a["symbol"]
            hist = get_history(sym, period="5d")
            cur = float(hist["Close"].iloc[-1]) if not hist.empty else None
            if cur is None:
                st.info(f"No price data for {sym}.")
                continue
            if a["dir"].startswith(">=") and cur >= a["price"] and not a["notified"]:
                st.success(f"ALERT: {sym} {a['dir']} {a['price']} (Now {cur})")
                a["notified"] = True
                any_alert = True
            if a["dir"].startswith("<=") and cur <= a["price"] and not a["notified"]:
                st.success(f"ALERT: {sym} {a['dir']} {a['price']} (Now {cur})")
                a["notified"] = True
                any_alert = True
        if not any_alert:
            st.info("No alerts triggered right now.")

# ---------- REPORTS ----------
with tabs[4]:
    st.header("Reports & Export")
    if st.button("Generate Report"):
        df = pd.DataFrame(st.session_state.positions)
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        if df.empty:
            st.warning("No positions to export.")
        else:
            md = f"# Portfolio Report — {now}\n\n" + df.to_markdown(index=False)
            st.download_button("Download Markdown", md, file_name="portfolio.md", mime="text/markdown")
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button("Download CSV", csv_buf.getvalue(), file_name="portfolio.csv", mime="text/csv")

st.markdown("---")
st.markdown("""
### Notes
- This app uses Google News RSS (no API keys). RSS is free but sometimes returns different results than paid news APIs.
- Summarization is local and heuristic-based: it extracts keywords, a representative headline, and a simple Bullish/Neutral/Bearish tone estimate.
- For lower latency or production needs, consider a streaming data provider (paid) or a background worker + DB. This current app remains API-free.
""")
