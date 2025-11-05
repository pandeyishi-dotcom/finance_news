# fin_news_fixed.py
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
    """
    Return dict: last, prev, info
    Safe: returns None for missing values instead of raising.
    """
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="5d")
        if hist is None or hist.empty:
            return {"last": None, "prev": None, "info": {}}
        # Ensure indices present
        close = hist["Close"].dropna()
        if close.empty:
            return {"last": None, "prev": None, "info": {}}
        last = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) > 1 else last
        # info can be heavy; use get to avoid KeyErrors
        try:
            info = t.info or {}
        except Exception:
            info = {}
        return {"last": last, "prev": prev, "info": info}
    except Exception:
        return {"last": None, "prev": None, "info": {}}

@st.cache_data(ttl=300)
def get_history(symbol, period="1y", interval="1d"):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period, interval=interval)
        if hist is None or hist.empty:
            return pd.DataFrame()
        return hist.reset_index()
    except Exception:
        return pd.DataFrame()

# ---- Live Tape ----
def render_tape():
    data = []
    for s in st.session_state.watchlist:
        p = get_price(s)
        last, prev = p["last"], p["prev"]
        if last is not None and prev is not None:
            try:
                pct = (last - prev) / prev * 100 if prev != 0 else 0.0
            except Exception:
                pct = 0.0
            arrow = "▲" if pct > 0 else ("▼" if pct < 0 else "")
            color = "green" if pct > 0 else ("red" if pct < 0 else "#9CA3AF")
            data.append(f"<b>{s}</b> {last:.2f} <span style='color:{color}'>{arrow} {pct:+.2f}%</span>")
        else:
            data.append(f"<b>{s}</b> -")
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

    # Show metrics in up to 4 columns
    cols = st.columns(min(4, max(1, len(st.session_state.watchlist))))
    for i, sym in enumerate(st.session_state.watchlist):
        data = get_price(sym)
        last, prev = data["last"], data["prev"]
        delta_display = "-"
        if last is not None and prev is not None:
            try:
                delta = (last - prev) / prev * 100 if prev != 0 else 0.0
                delta_display = f"{delta:+.2f}%"
            except Exception:
                delta_display = "-"
        cols[i % len(cols)].metric(sym, f"{last:.2f}" if last is not None else "-", delta_display)

    st.markdown("---")
    sel = st.selectbox("Select ticker", st.session_state.watchlist)
    hist = get_history(sel)
    if not hist.empty and "Date" in hist.columns and "Close" in hist.columns:
        st.line_chart(hist.set_index("Date")["Close"])
    else:
        st.warning("No history available for this symbol.")

# ========== NEWS & AI ==========
with tabs[1]:
    st.header("News & AI Insights")
    sym = st.selectbox("Select ticker", st.session_state.watchlist, key="news_sym")
    # yfinance news fallback
    news_items = []
    try:
        t = yf.Ticker(sym)
        # Some versions of yfinance return a list at .news, guard it
        raw_news = getattr(t, "news", None)
        if raw_news:
            for n in raw_news[:10]:
                title = n.get("title") if isinstance(n, dict) else str(n)
                publisher = n.get("publisher") if isinstance(n, dict) else ""
                link = n.get("link") if isinstance(n, dict) else ""
                news_items.append({"title": title, "publisher": publisher, "link": link})
    except Exception:
        news_items = []

    if news_items:
        for n in news_items:
            st.markdown(f"**{n.get('title','-')}**")
            if n.get("publisher"):
                st.caption(n.get("publisher"))
            if n.get("link"):
                st.write(n.get("link"))
            st.markdown("---")
    else:
        st.info("No recent news found via yfinance. Hook NewsAPI / vendor for richer feed.")

    # AI summary (optional)
    if OPENAI_INSTALLED and st.secrets.get("OPENAI_API_KEY"):
        if st.button("Generate AI Summary"):
            titles = "\n".join([n.get("title", "") for n in news_items]) or f"No headlines for {sym}"
            prompt = f"Summarize these headlines about {sym} into 3 concise bullets focused on market impact:\n{titles}"
            try:
                openai.api_key = st.secrets["OPENAI_API_KEY"]
                resp = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.2,
                )
                summary_text = resp.choices[0].text.strip() if hasattr(resp, "choices") else str(resp)
                st.subheader("🧠 AI Summary")
                st.write(summary_text)
            except Exception as e:
                st.error(f"AI summary failed: {e}")
    else:
        st.caption("Add OPENAI_API_KEY to Streamlit Secrets and install 'openai' to enable AI summaries.")

# ========== PORTFOLIO ==========
with tabs[2]:
    st.header("Portfolio Simulator")
    col1, col2, col3 = st.columns(3)
    sym_input = col1.text_input("Symbol")
    qty_input = col2.number_input("Qty", min_value=1, value=1)
    side_input = col3.selectbox("Side", ["long", "short"])
    if st.button("Add Position"):
        price = get_price(sym_input)["last"] or 0.0
        st.session_state.positions.append(
            {"symbol": sym_input.upper(), "qty": qty_input, "side": side_input, "entry": price, "created": datetime.utcnow().isoformat()}
        )

    if st.session_state.positions:
        df = pd.DataFrame(st.session_state.positions)
        rows = []
        total_pnl = 0.0
        for row in df.to_dict("records"):
            cur = get_price(row["symbol"])["last"] or 0.0
            pnl = (cur - row["entry"]) * row["qty"] * (1 if row["side"] == "long" else -1)
            rows.append(
                {
                    "Symbol": row["symbol"],
                    "Qty": row["qty"],
                    "Entry": row["entry"],
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
        info = get_price(s)["info"] or {}
        pe = info.get("trailingPE") or 9999
        mcap = info.get("marketCap") or 0
        if pe <= pe_limit and mcap >= mcap_limit:
            matches.append({"Symbol": s, "P/E": pe, "MarketCap": mcap})
    if matches:
        st.dataframe(pd.DataFrame(matches))
    else:
        st.info("No matches for current screener filters.")

# ========== BACKTEST ==========
with tabs[4]:
    st.header("Backtest: Moving Average Crossover (demo)")
    back_sym = st.selectbox("Select ticker", st.session_state.watchlist, key="bt_sym")
    short = st.number_input("Short MA", 5, 50, 20)
    long = st.number_input("Long MA", 10, 200, 50)
    hist = get_history(back_sym)
    if not hist.empty and "Close" in hist.columns:
        hist["SMA_Short"] = hist["Close"].rolling(short).mean()
        hist["SMA_Long"] = hist["Close"].rolling(long).mean()
        st.line_chart(hist.set_index("Date")[["Close", "SMA_Short", "SMA_Long"]])
        st.caption("Strategy: Buy when short MA crosses above long MA, sell when below.")
    else:
        st.warning("No historical data available for backtest.")

# ========== ALERTS ==========
with tabs[5]:
    st.header("Alerts")
    alert_sym = st.text_input("Symbol for alert", key="alert_sym")
    alert_price = st.number_input("Target price", 0.0, key="alert_price")
    direction = st.selectbox("Direction", [">= (cross above)", "<= (cross below)"], key="alert_dir")
    if st.button("Add Alert"):
        st.session_state.alerts.append({"symbol": alert_sym.upper(), "price": float(alert_price), "dir": direction, "notified": False, "created": datetime.utcnow().isoformat()})
        st.success("Alert added.")

    if st.session_state.alerts:
        st.dataframe(pd.DataFrame(st.session_state.alerts))
    else:
        st.info("No alerts created yet.")

    if st.button("Check Alerts"):
        triggered = []
        for a in st.session_state.alerts:
            cur = get_price(a["symbol"])["last"]
            if cur is None:
                continue
            if a["dir"].startswith(">=") and cur >= a["price"] and not a.get("notified"):
                triggered.append((a, cur))
                a["notified"] = True
            if a["dir"].startswith("<=") and cur <= a["price"] and not a.get("notified"):
                triggered.append((a, cur))
                a["notified"] = True
        if triggered:
            for tcur in triggered:
                a, curval = tcur
                st.success(f"ALERT: {a['symbol']} {a['dir']} {a['price']} — current {curval}")
        else:
            st.info("No alerts triggered at this time.")

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
        st.download_button("Download Markdown Report", md, file_name=f"portfolio_{datetime.utcnow().date()}.md")
        st.download_button("Download CSV", csv_buf.getvalue(), file_name=f"portfolio_{datetime.utcnow().date()}.csv")

# ---- Footer ----
st.markdown("---")
st.markdown(
    """
**Some text or table**
- point 1  
- point 2  
- etc...
""",
    unsafe_allow_html=True,
)

st.markdown(
    """,
    unsafe_allow_html=True,
)

### ✅ Next Steps
- Integrate Polygon / Finnhub for live data  
- Add persistent storage (Supabase / Firebase)  
- Integrate vectorbt or backtrader for pro backtests  
- Enable OpenAI in secrets for full AI functionality  

**Secrets Example (Streamlit Cloud):**
