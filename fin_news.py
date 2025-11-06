import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
from datetime import datetime
import time

# Optional OpenAI (if you want AI summaries)
try:
    import openai
    OPENAI_INSTALLED = True
except Exception:
    OPENAI_INSTALLED = False

st.set_page_config("Bloomberg-Lite News App", layout="wide", initial_sidebar_state="expanded")
st.title("📡 Bloomberg-Lite: News + Finance Terminal")

# Session state setup
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AAPL", "MSFT", "TSLA", "RELIANCE.NS"]
if "positions" not in st.session_state:
    st.session_state.positions = []
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# --- Utilities ---
@st.cache_data(ttl=300)
def get_history(sym, period="1y", interval="1d"):
    try:
        df = yf.Ticker(sym).history(period=period, interval=interval)
        if df is None:
            return pd.DataFrame()
        return df.reset_index()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_finance_news(api_key, query="finance OR investing", language="en", page_size=10):
    url = ("https://newsapi.org/v2/everything?"
           f"q={query}&language={language}&pageSize={page_size}&sortBy=publishedAt&apiKey={api_key}")
    try:
        r = requests.get(url, timeout=10)
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
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

# --- UI Tabs ---
tabs = st.tabs(["Market", "News Hub", "Portfolio", "Alerts", "Reports"])

# --- Market Tab ---
with tabs[0]:
    st.header("Market Overview")
    wl = st.text_area("Watchlist (comma separated)", ",".join(st.session_state.watchlist))
    if st.button("Update Watchlist"):
        st.session_state.watchlist = [s.strip().upper() for s in wl.split(",") if s.strip()]
        st.success("Watchlist updated.")
    sel = st.selectbox("Select ticker", st.session_state.watchlist)
    hist = get_history(sel)
    if not hist.empty:
        st.line_chart(hist.set_index("Date")["Close"])
    else:
        st.info("No data to show.")

# --- News Hub Tab ---
with tabs[1]:
    st.header("News Hub: Finance & Investing")
    if "NEWSAPI_KEY" in st.secrets:
        news_list = fetch_finance_news(st.secrets["NEWSAPI_KEY"])
        if news_list:
            for n in news_list:
                st.markdown(f"**{n['title']}**  \n*{n['source']} – {n['publishedAt'][:10]}*  \n[{n['url']}]({n['url']})")
                if n['description']:
                    st.markdown(f"_{n['description']}_")
                st.markdown("---")
        else:
            st.info("No news articles found.")
        if OPENAI_INSTALLED and st.button("Generate AI Summary"):
            titles = "\n".join([n["title"] for n in news_list[:5]])
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            resp = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"Summarize these finance headlines into 3 bullet points:\n{titles}",
                max_tokens=150, temperature=0.3
            )
            st.subheader("🧠 AI Summary")
            st.write(resp.choices[0].text.strip())
    else:
        st.warning("Add NEWSAPI_KEY in Streamlit Secrets to fetch live finance news.")

# --- Portfolio Tab ---
with tabs[2]:
    st.header("Portfolio Simulator")
    sym = st.text_input("Symbol to add")
    qty = st.number_input("Qty", min_value=1, value=1)
    side = st.selectbox("Side", ["long", "short"])
    if st.button("Add Position"):
        price = get_history(sym, period="5d")["Close"].iloc[-1] if not get_history(sym, period="5d").empty else 0.0
        st.session_state.positions.append({"symbol": sym.upper(), "qty": qty, "side": side, "entry": price})
        st.success("Position added.")
    if st.session_state.positions:
        df = pd.DataFrame(st.session_state.positions)
        st.dataframe(df)

# --- Alerts Tab ---
with tabs[3]:
    st.header("Alerts")
    a_sym = st.text_input("Symbol for alert", key="alert_sym")
    a_price = st.number_input("Price target", value=0.0, key="alert_price")
    direction = st.selectbox("Direction", [">= (cross above)", "<= (cross below)"], key="alert_dir")
    if st.button("Create Alert"):
        st.session_state.alerts.append({"symbol": a_sym.upper(), "price": float(a_price), "dir": direction, "notified": False})
        st.success("Alert created.")
    if st.session_state.alerts:
        st.dataframe(pd.DataFrame(st.session_state.alerts))
    if st.button("Check Alerts"):
        for a in st.session_state.alerts:
            cur = get_history(a["symbol"], period="5d")["Close"].iloc[-1] if not get_history(a["symbol"], period="5d").empty else None
            if cur is not None:
                if a["dir"].startswith(">=") and cur >= a["price"] and not a["notified"]:
                    st.success(f"ALERT: {a['symbol']} {a['dir']} {a['price']} (now {cur})")
                    a["notified"] = True
                if a["dir"].startswith("<=") and cur <= a["price"] and not a["notified"]:
                    st.success(f"ALERT: {a['symbol']} {a['dir']} {a['price']} (now {cur})")
                    a["notified"] = True

# --- Reports Tab ---
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

# Footer with instructions
st.markdown("---")
st.markdown(
    """
### ✅ Next Steps
- Add more filters for news: region, asset type  
- Integrate real-time tick data (Polygon / Finnhub)  
- Build persistent storage for positions & alerts (Supabase / Firebase)  
- Add advanced analytics & backtesting  
- If you add OPENAI key you enable AI summarization  
"""
)
# End of app
