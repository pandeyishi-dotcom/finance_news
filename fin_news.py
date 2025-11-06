import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
from datetime import datetime

# Optional OpenAI (works with both older openai and new OpenAI client)
OPENAI_INSTALLED = False
try:
    # new OpenAI client (pip package: openai >= ...)
    from openai import OpenAI as NewOpenAI
    NEW_OPENAI = True
    OPENAI_INSTALLED = True
except Exception:
    NEW_OPENAI = False
    try:
        import openai  # legacy client
        NEW_OPENAI = False
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

# ---------- UTILITIES ----------
@st.cache_data(ttl=300)
def get_history(sym: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Return history as a DataFrame with 'Date' column or empty DataFrame on failure."""
    sym = (sym or "").strip()
    if not sym:
        return pd.DataFrame()
    try:
        t = yf.Ticker(sym)
        df = t.history(period=period, interval=interval)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        # ensure column names are standard
        if "Date" not in df.columns and df.index.name is not None:
            df = df.reset_index()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_finance_news(api_key: str, query: str = "finance OR investing", language: str = "en", page_size: int = 10):
    """Fetch news from NewsAPI.org. Returns list of dicts or empty list."""
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
                "title": a.get("title") or "",
                "source": a.get("source", {}).get("name", ""),
                "publishedAt": a.get("publishedAt") or "",
                "url": a.get("url") or "",
                "description": a.get("description") or ""
            })
        return results
    except Exception as e:
        # don't call st.* inside cached function; return empty and let caller show message
        return []

# ---------- LAYOUT ----------
tabs = st.tabs(["Market", "News Hub", "Portfolio", "Alerts", "Reports"])

# ---------- MARKET TAB ----------
with tabs[0]:
    st.header("Market Overview")
    wl = st.text_area("Watchlist (comma separated)", ",".join(st.session_state.watchlist), height=80)
    if st.button("Update Watchlist"):
        st.session_state.watchlist = [s.strip().upper() for s in wl.split(",") if s.strip()]
        st.success("Watchlist updated.")
    if not st.session_state.watchlist:
        st.info("Watchlist empty. Add some tickers.")
    else:
        sel = st.selectbox("Select ticker", st.session_state.watchlist)
        hist = get_history(sel)
        if not hist.empty and "Close" in hist.columns:
            # use Date as index if present
            if "Date" in hist.columns:
                hist = hist.set_index("Date")
            st.line_chart(hist["Close"])
            st.write(hist.tail(5))
        else:
            st.info("No data to show for this ticker (invalid ticker or no recent data).")

# ---------- NEWS HUB TAB ----------
with tabs[1]:
    st.header("📰 Finance & Investing News")
    api_key = st.secrets.get("NEWSAPI_KEY") if hasattr(st, "secrets") else None
    if api_key:
        with st.spinner("Fetching news..."):
            news_list = fetch_finance_news(api_key)
        if news_list:
            for n in news_list:
                title = n.get("title", "")
                src = n.get("source", "")
                published = n.get("publishedAt", "")[:10]
                url = n.get("url", "")
                desc = n.get("description", "")
                st.markdown(f"**{title}**  \n*{src} — {published}*  \n[{url}]({url})")
                if desc:
                    st.markdown(f"_{desc}_")
                st.markdown("---")
        else:
            st.info("No recent finance news found or NewsAPI returned no articles.")
        # AI summary (if OpenAI available)
        if OPENAI_INSTALLED and st.secrets.get("OPENAI_API_KEY"):
            if st.button("🧠 Generate AI Summary of Headlines"):
                titles = "\n".join([n.get("title", "") for n in (news_list or [])][:8])
                if not titles.strip():
                    st.warning("Not enough headlines to summarize.")
                else:
                    try:
                        key = st.secrets["OPENAI_API_KEY"]
                        if NEW_OPENAI:
                            client = NewOpenAI(api_key=key)
                            # New client may present as chat.completions
                            resp = client.chat.completions.create(
                                model="gpt-4o-mini" if False else "gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are a financial journalist summarizing key market headlines."},
                                    {"role": "user", "content": f"Summarize these finance headlines into 3 concise bullet points:\n{titles}"}
                                ],
                                max_tokens=200,
                                temperature=0.4,
                            )
                            # best-effort extraction
                            summary = ""
                            if hasattr(resp, "choices") and len(resp.choices) > 0:
                                c = resp.choices[0]
                                if hasattr(c, "message"):
                                    summary = getattr(c.message, "content", "") or c.message.get("content", "")
                                else:
                                    summary = c.get("message", {}).get("content", "")
                        else:
                            # legacy openai package
                            openai.api_key = key
                            legacy_resp = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are a financial journalist summarizing key market headlines."},
                                    {"role": "user", "content": f"Summarize these finance headlines into 3 concise bullet points:\n{titles}"}
                                ],
                                max_tokens=200,
                                temperature=0.4,
                            )
                            summary = legacy_resp.choices[0].message["content"]
                        st.subheader("🧠 AI Summary")
                        st.write(summary)
                    except Exception as e:
                        st.error(f"OpenAI summarization failed: {e}")
        else:
            st.info("Add OPENAI_API_KEY to Streamlit secrets to enable AI summaries.")
    else:
        st.warning("Add NEWSAPI_KEY in Streamlit secrets to fetch live finance news.")

# ---------- PORTFOLIO TAB ----------
with tabs[2]:
    st.header("Portfolio Simulator")
    col1, col2, col3 = st.columns([3,1,1])
    with col1:
        sym = st.text_input("Symbol to add", "")
    with col2:
        qty = st.number_input("Qty", min_value=1, value=1, step=1)
    with col3:
        side = st.selectbox("Side", ["long", "short"])
    if st.button("Add Position"):
        sym = (sym or "").strip().upper()
        if not sym:
            st.error("Enter a valid symbol before adding.")
        else:
            hist = get_history(sym, period="5d")
            price = 0.0
            if not hist.empty and "Close" in hist.columns:
                try:
                    price = float(hist["Close"].iloc[-1])
                except Exception:
                    price = 0.0
            st.session_state.positions.append({"symbol": sym, "qty": int(qty), "side": side, "entry": float(price)})
            st.success("Position added.")
    if st.session_state.positions:
        df = pd.DataFrame(st.session_state.positions)
        st.dataframe(df)
    else:
        st.info("No positions yet. Add one above.")

# ---------- ALERTS TAB ----------
with tabs[3]:
    st.header("Price Alerts")
    a_sym = st.text_input("Symbol", key="alert_sym", value="")
    a_price = st.number_input("Target price", value=0.0, key="alert_price", format="%.4f")
    direction = st.selectbox("Direction", [">= (cross above)", "<= (cross below)"], key="alert_dir")
    if st.button("Create Alert"):
        if not a_sym.strip():
            st.error("Enter symbol to create alert.")
        else:
            st.session_state.alerts.append({
                "symbol": a_sym.strip().upper(),
                "price": float(a_price),
                "dir": direction,
                "notified": False
            })
            st.success("Alert created.")
    if st.session_state.alerts:
        st.dataframe(pd.DataFrame(st.session_state.alerts))
    else:
        st.info("No alerts configured.")

    if st.button("Check Alerts"):
        any_alert = False
        for a in st.session_state.alerts:
            sym = a.get("symbol")
            hist = get_history(sym, period="5d")
            cur = None
            if not hist.empty and "Close" in hist.columns:
                try:
                    cur = float(hist["Close"].iloc[-1])
                except Exception:
                    cur = None
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

# ---------- REPORTS TAB ----------
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

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("""
### ✅ Next Steps
- Add India RSS feeds (Moneycontrol, Economic Times, LiveMint) and parse feeds with `feedparser`.
- Integrate Polygon/Finnhub/AlphaVantage for real-time quotes (requires paid keys for true real-time).
- Store positions/alerts in Supabase/Postgres for persistence (instead of session_state).
- Enhance AI to give sentiment and market mood summaries (use embeddings + few-shot prompts).
- Add error logging and unit tests.
""")
