import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta
import io
import json
import re

# Optional AI import
try:
    import openai
    OPENAI_INSTALLED = True
except Exception:
    OPENAI_INSTALLED = False

# ---- Page setup ----
st.set_page_config(page_title="Bloomberg-Lite v3", layout="wide", initial_sidebar_state="expanded")

# ---- Theme toggle ----
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if st.session_state.dark_mode:
    st.markdown(
        """
        <style>
        body {background-color:#0a0b0c; color:#e6eef3;}
        .stMarkdown {color:#e6eef3;}
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        body {background-color:#f8fafc; color:#0a0b0c;}
        .stMarkdown {color:#0a0b0c;}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---- Session state defaults ----
defaults = {
    "watchlist": ["AAPL", "MSFT", "TSLA", "RELIANCE.NS"],
    "positions": [],
    "alerts": [],
    "live": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---- Header ----
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title("📊 Bloomberg-Lite v3 — Advanced Prototype")
with col2:
    st.session_state.live = st.checkbox("Live Tape", value=st.session_state.live)
with col3:
    st.session_state.dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode)

# ---- Utilities ----
@st.cache_data(ttl=30)
def get_price(symbol):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="5d")
        last = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2] if len(hist) > 1 else last
        info = t.info
        return {"last": last, "prev": prev, "info": info}
    except Exception:
        return {"last": None, "prev": None, "info": {}}

@st.cache_data(ttl=300)
def get_history(symbol, period="1y", interval="1d"):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period, interval=interval)
        return hist.reset_index()
    except Exception:
        return pd.DataFrame()

# ---- Live Tape ----
def render_tape():
    data = []
    for s in st.session_state.watchlist:
        p = get_price(s)
        last, prev = p["last"], p["prev"]
        if last and prev:
            pct = (last - prev) / prev * 100
            arrow = "▲" if pct > 0 else "▼"
            color = "green" if pct > 0 else "red"
            data.append(f"<b>{s}</b> {last:.2f} <span style='color:{color}'>{arrow} {pct:+.2f}%</span>")
    html = " &nbsp; | &nbsp; ".join(data)
    st.markdown(f"<div style='padding:8px;background:#111827;border-radius:6px'>{html}</div>", unsafe_allow_html=True)

if st.session_state.live:
    render_tape()

# ---- Tabs ----
tabs = st.tabs(
    [
        "Market",
        "News & AI",
        "Portfolio",
        "Screener",
        "Backtest",
        "Alerts",
        "Reports",
    ]
)

# ========== MARKET TAB ==========
with tabs[0]:
    st.header("Market Overview")
    wl = st.text_area("Watchlist (comma-separated)", ",".join(st.session_state.watchlist))
    if st.button("Update Watchlist"):
        st.session_state.watchlist = [x.strip().upper() for x in wl.split(",") if x.strip()]
        st.success("Watchlist updated.")

    cols = st.columns(min(4, len(st.session_state.watchlist)))
    for i, sym in enumerate(st.session_state.watchlist):
        data = get_price(sym)
        last, prev = data["last"], data["prev"]
        delta = (last - prev) / prev * 100 if last and prev else 0
        cols[i % 4].metric(sym, f"{last:.2f}" if last else "-", f"{delta:+.2f}%")

    st.markdown("---")
    sel = st.selectbox("Select ticker", st.session_state.watchlist)
    hist = get_history(sel)
    if not hist.empty:
        st.line_chart(hist.set_index("Date")["Close"])
    else:
        st.warning("No data available for this symbol.")

# ========== NEWS & AI ==========
with tabs[1]:
    st.header("News & AI Insights")
    sym = st.selectbox("Select ticker", st.session_state.watchlist, key="news_sym")
    try:
        news_items = yf.Ticker(sym).news[:10]
    except Exception:
        news_items = []
    if news_items:
        for n in news_items:
            st.markdown(f"**{n['title']}**  \n*{n['publisher']}*  \n[{n['link']}]({n['link']})")
            st.markdown("---")
    else:
        st.info("No recent news found.")

    if OPENAI_INSTALLED and "OPENAI_API_KEY" in st.secrets:
        if st.button("Generate AI Summary"):
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            titles = "\n".join([n["title"] for n in news_items])
            prompt = f"Summarize these headlines about {sym} into short bullet points of market impact:\n{titles}"
            try:
                resp = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=180,
                    temperature=0.3,
                )
                st.subheader("🧠 AI Summary")
                st.write(resp.choices[0].text.strip())
            except Exception as e:
                st.error(f"AI summary failed: {e}")
    else:
        st.caption("Add OPENAI_API_KEY in secrets to enable AI summaries.")

# ========== PORTFOLIO ==========
with tabs[2]:
    st.header("Portfolio Simulator")
    col1, col2, col3 = st.columns(3)
    sym = col1.text_input("Symbol")
    qty = col2.number_input("Qty", min_value=1, value=1)
    side = col3.selectbox("Side", ["long", "short"])
    if st.button("Add Position"):
        price = get_price(sym)["last"] or 0.0
        st.session_state.positions.append(
            {"symbol": sym.upper(), "qty": qty, "side": side, "entry": price}
        )
    if st.session_state.positions:
        df = pd.DataFrame(st.session_state.positions)
        rows = []
        total_pnl = 0
        for row in df.itertuples():
            cur = get_price(row.symbol)["last"] or 0
            pnl = (cur - row.entry) * row.qty * (1 if row.side == "long" else -1)
            rows.append(
                {
                    "Symbol": row.symbol,
                    "Qty": row.qty,
                    "Entry": row.entry,
                    "Current": cur,
                    "PnL": pnl,
                }
            )
            total_pnl += pnl
        dfp = pd.DataFrame(rows)
        st.metric("Total Unrealized P&L", f"₹{total_pnl:,.2f}")
        st.dataframe(dfp)

# ========== SCREENER ==========
with tabs[3]:
    st.header("Screener")
    pe_limit = st.slider("Max P/E", 5, 200, 50)
    mcap_limit = st.number_input("Min MarketCap", 0, 1_000_000_000_000, 0)
    matches = []
    for s in st.session_state.watchlist:
        info = get_price(s)["info"]
        pe = info.get("trailingPE") or 9999
        mcap = info.get("marketCap") or 0
        if pe <= pe_limit and mcap >= mcap_limit:
            matches.append({"Symbol": s, "P/E": pe, "MarketCap": mcap})
    st.dataframe(pd.DataFrame(matches))

# ========== BACKTEST ==========
with tabs[4]:
    st.header("Backtest: Moving Average Crossover")
    sym = st.selectbox("Select ticker", st.session_state.watchlist, key="bt_sym")
    short = st.number_input("Short MA", 5, 50, 20)
    long = st.number_input("Long MA", 10, 200, 50)
    hist = get_history(sym)
    if not hist.empty:
        hist["SMA_Short"] = hist["Close"].rolling(short).mean()
        hist["SMA_Long"] = hist["Close"].rolling(long).mean()
        st.line_chart(hist.set_index("Date")[["Close", "SMA_Short", "SMA_Long"]])
        st.caption("Strategy: Buy when short MA crosses above long MA, sell when below.")
    else:
        st.warning("No data available.")

# ========== ALERTS ==========
with tabs[5]:
    st.header("Alerts")
    sym = st.text_input("Symbol for alert")
    price = st.number_input("Target price", 0.0)
    direction = st.selectbox("Direction", [">= (cross above)", "<= (cross below)"])
    if st.button("Add Alert"):
        st.session_state.alerts.append(
            {"symbol": sym.upper(), "price": price, "dir": direction, "notified": False}
        )
        st.success("Alert added.")

    st.dataframe(pd.DataFrame(st.session_state.alerts))

    if st.button("Check Alerts"):
        for a in st.session_state.alerts:
            cur = get_price(a["symbol"])["last"]
            if not cur:
                continue
            if (
                a["dir"].startswith(">=")
                and cur >= a["price"]
                and not a["notified"]
            ) or (
                a["dir"].startswith("<=")
                and cur <= a["price"]
                and not a["notified"]
            ):
                st.success(f"ALERT: {a['symbol']} {a['dir']} {a['price']} (Now {cur})")
                a["notified"] = True

# ========== REPORTS ==========
with tabs[6]:
    st.header("Reports & Exports")
    if st.button("Generate Portfolio Report"):
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        df = pd.DataFrame(st.session_state.positions)
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        md = f"# Portfolio Report — {now}\n\n"
        md += df.to_markdown(index=False)
        st.download_button(
            "Download Markdown Report",
            md,
            file_name=f"portfolio_{datetime.utcnow().date()}.md",
        )
        st.download_button(
            "Download CSV",
            csv_buf.getvalue(),
            file_name=f"portfolio_{datetime.utcnow().date()}.csv",
        )

# ---- Footer ----
st.markdown("---")
st.markdown(
    st.markdown(
    """
**Some text or table**
- point 1  
- point 2  
- etc...
""",
    unsafe_allow_html=True,
)

### ✅ Next Steps
- Integrate Polygon / Finnhub for live data  
- Add persistent storage (Supabase / Firebase)  
- Integrate vectorbt or backtrader for pro backtests  
- Enable OpenAI in secrets for full AI functionality  

**Secrets Example (Streamlit Cloud):**
