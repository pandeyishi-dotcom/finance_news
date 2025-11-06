# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
from datetime import datetime
import time
import traceback

# Optional: auto refresh
try:
    from streamlit_autorefresh import st_autorefresh
    AUTORELOAD_AVAILABLE = True
except Exception:
    AUTORELOAD_AVAILABLE = False

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Bloomberg-Lite — Gemini Summaries", layout="wide", initial_sidebar_state="expanded")
st.title("📡 Bloomberg-Lite — Gemini (Google) Summaries & Market Dashboard")

# ---------- SECRETS / CONFIG ----------
# Put your Google GenAI API key in Streamlit secrets with key name: GOOGLE_API_KEY
# Example .streamlit/secrets.toml:
# GOOGLE_API_KEY = "AIza...."   (or the key you received in Google Cloud/GenAI)
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, "secrets") else None

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
def get_history(sym: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
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

def get_latest_price(sym: str):
    """Best-effort fetch of latest intraday price; returns float or None."""
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

@st.cache_data(ttl=300)
def fetch_finance_news(api_key: str, query: str="finance OR investing", language: str="en", page_size: int=10):
    """Fetch from NewsAPI.org (if you store NEWSAPI_KEY) — optional."""
    if not api_key:
        return []
    try:
        url = (
            "https://newsapi.org/v2/everything?"
            f"q={requests.utils.quote(query)}&language={language}&pageSize={page_size}&sortBy=publishedAt&apiKey={api_key}"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        articles = j.get("articles", [])
        out = []
        for a in articles:
            out.append({
                "title": a.get("title") or "",
                "source": (a.get("source") or {}).get("name", ""),
                "publishedAt": a.get("publishedAt") or "",
                "url": a.get("url") or "",
                "description": a.get("description") or ""
            })
        return out
    except Exception:
        return []

# ---------- Gemini summarizer ----------
def gemini_summarize(text: str, model: str = "gemini-1.5"):
    """
    Generate a short 3-bullet summary using Google Gemini (GenAI).
    This function tries to use the google genai client (preferred).
    If that is not present or fails, it raises an Exception that the caller should catch.
    """
    # Minimal validation
    if not text or not GOOGLE_API_KEY:
        raise RuntimeError("Missing text or GOOGLE_API_KEY for Gemini summarization.")

    # Try the official google genai client (recommended)
    try:
        # Try import patterns used in official docs
        try:
            # modern unified SDK
            from google import genai
            # instantiate client and call generate
            client = genai.Client(api_key=GOOGLE_API_KEY) if hasattr(genai, "Client") else None
            if client is not None and hasattr(client, "models") and hasattr(client.models, "generate_content"):
                # recommended high-level API
                resp = client.models.generate_content(model=model, contents=text, temperature=0.2, top_k=40)
                # response may expose .text or .output or .candidates
                if hasattr(resp, "text"):
                    return resp.text
                # fallback dict parsing
                try:
                    return resp["candidates"][0]["content"]
                except Exception:
                    return str(resp)
            # older API surface
            if hasattr(genai, "generate") or hasattr(genai, "generate_text"):
                # attempt to call legacy API surface
                if hasattr(genai, "generate"):
                    out = genai.generate(model=model, prompt=text)
                else:
                    out = genai.generate_text(model=model, prompt=text)
                # out parsing
                if isinstance(out, dict) and "candidates" in out and len(out["candidates"])>0:
                    return out["candidates"][0].get("content", "")
                if hasattr(out, "text"):
                    return out.text
                return str(out)
        except Exception:
            # If import-from-google fails, try google-genai package name
            pass

        # Alternative import path used by some official samples:
        try:
            import google.genai as genai2  # newer package layout seen in some docs
            client = genai2.Client(api_key=GOOGLE_API_KEY) if hasattr(genai2, "Client") else None
            if client and hasattr(client.models, "generate_content"):
                resp = client.models.generate_content(model=model, contents=text, temperature=0.2, top_k=40)
                if hasattr(resp, "text"):
                    return resp.text
                try:
                    return resp["candidates"][0]["content"]
                except Exception:
                    return str(resp)
        except Exception:
            pass

        # If none succeeded, raise to let caller fallback
        raise RuntimeError("GenAI client not available or failed to run.")
    except Exception as e:
        # bubble up the original stack for clearer logging in the app
        raise

def safe_summary(headlines_text: str):
    """A safe, simple fallback summarizer (no external calls)."""
    # Produce 3 concise bullets by picking top 3 headlines and truncating
    lines = [l.strip() for l in headlines_text.splitlines() if l.strip()]
    bullets = []
    for i, line in enumerate(lines[:6]):  # take first 6 candidate lines
        # simple heuristics: pick the headlines that contain finance keywords
        bullets.append("• " + (line[:140] + ("…" if len(line) > 140 else "")))
        if len(bullets) >= 3:
            break
    if not bullets:
        return "No headlines to summarize."
    return "\n".join(bullets)

# ---------- LAYOUT ----------
tabs = st.tabs(["Market", "News Hub", "Portfolio", "Alerts", "Reports"])

# ---------- MARKET TAB ----------
with tabs[0]:
    st.header("Market Overview")
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

    # update live prices when polling active (best effort)
    if st.session_state.polling:
        for s in st.session_state.watchlist:
            price = get_latest_price(s)
            if price is not None:
                st.session_state.live_prices[s] = {"price": price, "ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
    # display live prices
    if st.session_state.live_prices:
        rows = []
        for s in st.session_state.watchlist:
            info = st.session_state.live_prices.get(s)
            rows.append({
                "symbol": s,
                "last_price": info["price"] if info else None,
                "updated_at": info["ts"] if info else None
            })
        st.dataframe(pd.DataFrame(rows))
    else:
        st.info("No live prices yet. Start polling or click 'Refresh prices now'.")

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
    st.header("📰 Finance & Investing News")
    # Optional NewsAPI key support (add NEWSAPI_KEY to secrets if you want)
    NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY") if hasattr(st, "secrets") else None
    # Fetch news from NewsAPI if key present, else fallback to Google News RSS
    news_list = []
    if NEWSAPI_KEY:
        news_list = fetch_finance_news(NEWSAPI_KEY, page_size=10)
    else:
        # fallback: Google News RSS
        try:
            import feedparser
            rss_url = "https://news.google.com/rss/search?q=finance+OR+markets+OR+stocks&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(rss_url)
            for e in feed.entries[:10]:
                news_list.append({
                    "title": getattr(e, "title", ""),
                    "source": getattr(e, "source", {}).get("title", "") if hasattr(e, "source") else "",
                    "publishedAt": getattr(e, "published", ""),
                    "url": getattr(e, "link", ""),
                    "description": getattr(e, "summary", "")
                })
        except Exception:
            # If feedparser missing, simple no-news message
            pass

    if news_list:
        for n in news_list:
            st.markdown(f"**{n['title']}**  \n*{n['source']} — {n['publishedAt'][:10]}*  \n[{n['url']}]({n['url']})")
            if n.get("description"):
                st.markdown(f"_{n['description']}_")
            st.markdown("---")
    else:
        st.info("No recent news found (add NEWSAPI_KEY to secrets for NewsAPI support).")

    st.markdown("### AI Summary (Gemini)")
    if not GOOGLE_API_KEY:
        st.warning("Add GOOGLE_API_KEY to Streamlit secrets to enable Gemini summarization.")
    else:
        if st.button("🧠 Generate Gemini Summary of Headlines"):
            # prepare headlines text
            titles = "\n".join([n["title"] for n in news_list[:12]]) if news_list else ""
            if not titles.strip():
                st.warning("No headlines found to summarize.")
            else:
                with st.spinner("Generating summary via Gemini..."):
                    try:
                        # call gemini summarizer
                        # craft a concise instruction
                        prompt = (
                            "You are a helpful financial journalist. Summarize the following finance headlines into "
                            "3 concise bullet points (each 1 short sentence) that reflect the main market theme and tone:\n\n"
                            f"{titles}\n\nOutput only the three bullets (prefix each bullet with a dash or number)."
                        )
                        # Use gemini_summarize helper
                        summary_text = gemini_summarize(prompt, model="gemini-1.5")
                        if not summary_text:
                            raise RuntimeError("Empty response from Gemini.")
                        st.subheader("🧠 Gemini Summary")
                        st.write(summary_text)
                    except Exception as e:
                        # log error to Streamlit app (no secrets printed)
                        st.error("Gemini summarization failed — showing a safe fallback summary.")
                        st.write(f"*Error (hidden)*: {type(e).__name__} — {str(e)[:120]}")
                        # fallback summary
                        st.subheader("Fallback summary (no Gemini)")
                        st.write(safe_summary(titles))

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
    a_price = st.number_input("Target price", value=0.0, key="alert_price", format="%.4f")
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

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("""
### ✅ Next steps & notes
- Set `GOOGLE_API_KEY` in Streamlit secrets (see README instructions below).
- Gemini (GenAI) may require you to enable Vertex AI / GenAI API in a Google Cloud project and/or use the appropriate API key type. If the GenAI client raises an error, you will see a fallback summary.
- For true production-grade streaming use Polygon/Finnhub + a worker process + Redis/Supabase.
""")
