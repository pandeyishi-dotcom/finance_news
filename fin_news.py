# app_v3.py
"""
Bloomberg-Lite Pro v3 — All-features prototype (single file)
- Multi-tab advanced layout
- Live tape, dark mode, AI news summarizer (optional)
- Portfolio, Screener, Backtest, Alerts, Reports
- Uses yfinance for free data; optional: set OPENAI_API_KEY, POLYGON_API_KEY, NEWSAPI_KEY in Streamlit secrets
Run: streamlit run app_v3.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta
import io
import json
import re

# Optional imports (guarded)
try:
    import openai
    OPENAI_INSTALLED = True
except Exception:
    OPENAI_INSTALLED = False

# ---- Page / theme ----
st.set_page_config(page_title="Bloomberg-Lite v3", layout="wide", initial_sidebar_state="expanded")

# CSS: simple dark/light styles (toggle below)
BASE_CSS = """
<style>
body { background-color: #0a0b0c; color: #e6eef3; }
.header { display:flex; align-items:center; gap:12px; }
.kbd { background:#111827; padding:4px 6px; border-radius:4px; color:#fff; }
.cmd { background:#071018; padding:8px; border-radius:8px; }
.small { color:#9CA3AF; font-size:12px; }
.card { background:#071018; padding:10px; border-radius:8px; }
</style>
"""

LIGHT_CSS = """
<style>
body { background-color: #f7f7f8; color: #0b1020; }
.header { display:flex; align-items:center; gap:12px; }
.kbd { background:#e6eef3; padding:4px 6px; border-radius:4px; color:#0b1020; }
.cmd { background:#ffffff; padding:8px; border-radius:8px; }
.small { color:#4b5563; font-size:12px; }
.card { background:#ffffff; padding:10px; border-radius:8px; }
</style>
"""

# session defaults
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AAPL", "MSFT", "TSLA", "RELIANCE.NS"]
if "live" not in st.session_state:
    st.session_state.live = False
if "positions" not in st.session_state:
    st.session_state.positions = []  # list of dicts
if "alerts" not in st.session_state:
    st.session_state.alerts = []  # list of dicts
if "last_fetch" not in st.session_state:
    st.session_state.last_fetch = {}
if "command_history" not in st.session_state:
    st.session_state.command_history = []

# apply CSS
if st.session_state.dark_mode:
    st.markdown(BASE_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

# ---- Top header and command palette ----
header_cols = st.columns([3,2,1])
with header_cols[0]:
    st.markdown("<div class='header'><h2 style='margin:0'>Bloomberg-Lite v3</h2><div class='small'> — Multi-tool Terminal Prototype</div></div>", unsafe_allow_html=True)
with header_cols[1]:
    # live tape controls and theme
    st.session_state.live = st.checkbox("Live Tape", value=st.session_state.live, key="live_tape")
    st.session_state.dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode, key="theme_toggle")
with header_cols[2]:
    st.markdown("<div style='text-align:right'><span class='small'>Ctrl+K to focus command</span></div>", unsafe_allow_html=True)

# command input (global)
cmd = st.text_input("Command (e.g., ADD AAPL | AAPL GO | SCREENER pe<20 )", key="global_cmd")
if st.button("Run", key="cmd_run"):
    # store history
    st.session_state.command_history.append({"cmd": cmd, "ts": datetime.utcnow().isoformat()})
    # parse simple commands
    tok = cmd.strip().split()
    if not tok:
        st.info("Empty command.")
    else:
        verb = tok[0].lower()
        args = tok[1:]
        if verb in ("add", "addticker"):
            if args:
                sym = args[0].upper()
                if sym not in st.session_state.watchlist:
                    st.session_state.watchlist.append(sym)
                    st.success(f"Added {sym}")
                else:
                    st.info(f"{sym} already in watchlist")
        elif verb in ("remove", "rm"):
            if args:
                sym = args[0].upper()
                st.session_state.watchlist = [t for t in st.session_state.watchlist if t != sym]
                st.success(f"Removed {sym}")
        elif verb in ("screener", "screen"):
            # store screener expression
            st.session_state.last_screener = " ".join(args)
            st.success("Screener query saved (use Screener tab to run).")
        else:
            # treat as ticker open
            candidate = tok[0].upper()
            if re.match(r"^[A-Z0-9\.\-:]+$", candidate):
                st.session_state.open_ticker = candidate
                st.success(f"Selected {candidate}")
            else:
                st.info("Command registered.")

# inject JS for Ctrl+K focusing command input
st.components.v1.html("""
<script>
document.addEventListener('keydown', function(e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    const el = window.parent.document.querySelector('input[aria-label="Command (e.g., ADD AAPL | AAPL GO | SCREENER pe<20 )"]');
    if (el) { el.focus(); e.preventDefault(); }
  }
});
</script>
""", height=0)

# ---- Simple data fetch utilities (yfinance fallback) ----
@st.cache_data(ttl=30)
def fetch_history_yf(ticker: str, period="1y", interval="1d"):
    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval=interval)
    return hist.reset_index() if not hist.empty else pd.DataFrame()

@st.cache_data(ttl=15)
def fetch_price_yf(ticker: str):
    try:
        t = yf.Ticker(ticker)
        h = t.history(period="5d", interval="1d")
        last = None
        prev = None
        if not h.empty:
            last = float(h["Close"].iloc[-1])
            prev = float(h["Close"].iloc[-2]) if len(h) > 1 else last
        info = t.info or {}
        return {"symbol": ticker, "last": last, "prev": prev, "info": info}
    except Exception as e:
        return {"symbol": ticker, "error": str(e)}

# Placeholder for advanced provider hooks (Polygon / Finnhub)
# TODO: create fetch_price_polygon / fetch_ohlcv_polygon and point code at them when POLYGON_API_KEY in secrets

# ---- helper functions ----
def human(x):
    if x is None: return "-"
    for unit in ["", "K","M","B","T"]:
        if abs(x) < 1000:
            return f"{x:.2f}{unit}"
        x /= 1000.0
    return f"{x:.2f}Q"

def run_screener_expr(tickers, expr):
    """
    Very simple screener executor: expr examples:
      'pe<20', 'pe<20 and marketcap>1e10'
    It uses yfinance info fields (trailingPE, marketCap).
    WARNING: simple eval on constructed dict; sanitized by allowed keys only.
    """
    if not expr:
        return pd.DataFrame()
    results = []
    for t in tickers:
        info = fetch_price_yf(t).get("info", {})
        context = {
            "pe": info.get("trailingPE") or 999999,
            "marketcap": info.get("marketCap") or 0,
            "divy": info.get("dividendYield") or 0
        }
        try:
            # safe eval: only allow numbers and basic operators by using regex replace of allowed names
            allowed = {"pe","marketcap","divy"}
            # very simple safety: forbid letters other than allowed names and operators
            if re.search(r"[^0-9<>=.&|() a-zA-Z_+-/*]", expr):
                continue
            ok = eval(expr, {"__builtins__":None}, context)
            if ok:
                results.append({"symbol": t, "pe": context["pe"], "marketcap": context["marketcap"], "dividend": context["divy"]})
        except Exception:
            continue
    return pd.DataFrame(results)

# ---- Top live tape area ----
def render_tape(tickers):
    # produce small inline "tape" html
    parts = []
    for s in tickers:
        p = fetch_price_yf(s)
        last = p.get("last")
        prev = p.get("prev") or last
        pct = None
        if last is not None and prev:
            pct = (last - prev)/prev*100
        arrow = "▲" if pct and pct>0 else ("▼" if pct and pct<0 else "")
        pct_str = f"{pct:+.2f}%" if pct is not None else ""
        parts.append(f"<b>{s}</b> {last or '-'} <small style='color:#9CA3AF'>{arrow} {pct_str}</small>")
    html = "<div style='padding:8px;border-radius:6px;background:#071018;overflow:auto'>" + " &nbsp; &nbsp; ".join(parts) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

if st.session_state.live:
    render_tape(st.session_state.watchlist)
else:
    # condensed static tape
    render_tape(st.session_state.watchlist[:8])

# ---- Tabs: Market / News / Portfolio / Screener / Backtest / Alerts / Reports ----
tabs = st.tabs(["Market","News & AI","Portfolio","Screener","Backtest","Alerts","Reports"])

# --- MARKET TAB ---
with tabs[0]:
    st.header("Market Overview")
    # Watchlist editor
    wl = st.text_area("Watchlist (comma separated)", value=",".join(st.session_state.watchlist), height=80)
    if st.button("Save Watchlist", key="save_wl"):
        st.session_state.watchlist = [x.strip().upper() for x in wl.split(",") if x.strip()]
        st.success("Watchlist updated.")
    # Grid of metric cards
    cols = st.columns(min(4, max(1, len(st.session_state.watchlist))))
    for i, sym in enumerate(st.session_state.watchlist):
        col = cols[i % len(cols)]
        p = fetch_price_yf(sym)
        last = p.get("last")
        prev = p.get("prev") or last
        pct = None
        if last is not None and prev:
            pct = (last - prev)/prev*100
        label = f"{sym}"
        col.metric(label=label, value=f"{last}", delta=f"{pct:+.2f}%" if pct is not None else "")
    st.markdown("---")
    # Selected ticker detail
    sel = st.selectbox("Open ticker", options=st.session_state.watchlist, index=0, key="market_select")
    hist = fetch_history_yf(sel, period="1y", interval="1d")
    if not hist.empty:
        st.line_chart(hist.set_index("Date")["Close"])
        # fundamentals snapshot
        info = fetch_price_yf(sel).get("info", {})
        st.write("Fundamentals (quick):", {"MarketCap": human(info.get("marketCap")), "PE": info.get("trailingPE"), "DividendYield": info.get("dividendYield")})
    else:
        st.info("No history available for this symbol (yfinance fallback).")

# --- NEWS & AI TAB ---
with tabs[1]:
    st.header("News & AI Insights")
    st.markdown("**AI News Summarizer** (optional): summarises headlines for selected ticker using OpenAI if enabled.")
    news_ticker = st.selectbox("Ticker for news", options=st.session_state.watchlist, index=0, key="news_ticker")
    # fetch news via yfinance (if available) or use NewsAPI if NEWSAPI_KEY in secrets
    news_items = []
    try:
        nf = yf.Ticker(news_ticker).news or []
        for n in nf[:12]:
            news_items.append({"title": n.get("title"), "publisher": n.get("publisher"), "link": n.get("link")})
    except Exception:
        news_items = []
    # Optionally use NewsAPI (if key provided)
    if "NEWSAPI_KEY" in st.secrets:
        # TODO: integrate NewsAPI fetch. For now, keep yfinance headlines.
        pass

    if news_items:
        for n in news_items:
            st.markdown(f"**{n['title']}**")
            st.caption(n.get("publisher") or "")
            if n.get("link"):
                st.write(n["link"])
            st.markdown("---")
    else:
        st.info("No news via yfinance. Hook NewsAPI or a paid vendor for robust feed.")

    # AI summarizer
    ai_enabled = st.checkbox("Enable AI Summaries (requires OPENAI_API_KEY)", value=False)
    if ai_enabled:
        if OPENAI_INSTALLED and st.secrets.get("OPENAI_API_KEY"):
            try:
                openai.api_key = st.secrets["OPENAI_API_KEY"]
                # combine titles
                headlines = "\n".join([n["title"] for n in news_items[:6]]) or f"No headlines found for {news_ticker}"
                prompt = f"Summarize the following headlines into 4 concise bullets focused on market impact:\n\n{headlines}"
                # safe, short completion
                resp = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=180, temperature=0.2)
                summary = resp.choices[0].text.strip()
                st.markdown("### AI Summary")
                st.write(summary)
            except Exception as e:
                st.error("AI call failed: " + str(e))
        else:
            st.warning("OpenAI library/key not configured. Add OPENAI_API_KEY to Streamlit secrets and include openai in requirements to enable.")

# --- PORTFOLIO TAB ---
with tabs[2]:
    st.header("Portfolio & Paper Trading")
    st.markdown("Create / manage a simple paper portfolio. Positions are stored in session only.")
    # Add position
    p1, p2, p3 = st.columns([2,1,1])
    with p1:
        add_sym = st.text_input("Ticker to add", value="", key="add_sym")
    with p2:
        add_qty = st.number_input("Qty", min_value=1, value=1, step=1, key="add_qty")
    with p3:
        add_side = st.selectbox("Side", ["long","short"], key="add_side")
    if st.button("Add Position"):
        price = fetch_price_yf(add_sym).get("last") or 0.0
        pos = {"symbol": add_sym.upper(), "qty": float(add_qty), "side": add_side, "entry": float(price), "created": datetime.utcnow().isoformat()}
        st.session_state.positions.append(pos)
        st.success(f"Added {pos['qty']} {pos['symbol']} ({pos['side']}) @ {pos['entry']:.2f}")

    if st.session_state.positions:
        dfp = pd.DataFrame(st.session_state.positions)
        st.table(dfp)
        # portfolio summary
        total = 0.0
        rows = []
        for row in st.session_state.positions:
            cur = fetch_price_yf(row["symbol"]).get("last") or 0.0
            pl = (cur - row["entry"])*row["qty"]*(1 if row["side"]=="long" else -1)
            rows.append({"symbol": row["symbol"], "qty": row["qty"], "entry": row["entry"], "current": cur, "pnl": pl})
            total += pl
        st.metric("Unrealized P&L", f"₹{total:.2f}")
        st.dataframe(pd.DataFrame(rows))

# --- SCREENER TAB ---
with tabs[3]:
    st.header("Screener (classic + NLP)")
    st.markdown("You can use sliders or type queries like `pe<20 and marketcap>1e10` in the box and click Run.")
    # sliders (common)
    max_pe = st.slider("Max P/E", 5, 200, 80)
    min_mcap = st.number_input("Min MarketCap (absolute)", value=0)
    expr = st.text_input("Advanced expression (e.g. pe<20 and marketcap>1e10)", value=st.session_state.get("last_screener",""))
    if st.button("Run Screener (expr)"):
        dfres = run_screener_expr(st.session_state.watchlist, expr)
        if dfres.empty:
            st.info("No matches or expr invalid.")
        else:
            st.dataframe(dfres)
    if st.button("Run slider screener"):
        matches = []
        for s in st.session_state.watchlist:
            info = fetch_price_yf(s).get("info",{})
            pe = info.get("trailingPE") or 999999
            mcap = info.get("marketCap") or 0
            if pe <= max_pe and mcap >= min_mcap:
                matches.append({"symbol": s, "pe": pe, "marketcap": mcap})
        st.table(pd.DataFrame(matches))

# --- BACKTEST TAB ---
with tabs[4]:
    st.header("Backtest Lab — MA Crossover (demo)")
    bt_sym = st.selectbox("Symbol", options=st.session_state.watchlist, index=0, key="bt_sym")
    short_ma = st.number_input("Short MA (days)", min_value=1, value=20, key="bt_short")
    long_ma = st.number_input("Long MA (days)", min_value=1, value=50, key="bt_long")
    capital = st.number_input("Initial capital", value=100000, step=10000)
    if st.button("Run Backtest"):
        hist = fetch_history_yf(bt_sym, period="2y", interval="1d")
        if hist.empty:
            st.error("No data for backtest.")
        else:
            df = hist.copy()
            df["short"] = df["Close"].rolling(short_ma).mean()
            df["long"] = df["Close"].rolling(long_ma).mean()
            df["signal"] = 0
            df.loc[df["short"] > df["long"], "signal"] = 1
            df.loc[df["short"] <= df["long"], "signal"] = 0
            df["positions"] = df["signal"].diff().fillna(0)
            cash = capital
            shares = 0
            equity_curve = []
            for idx, row in df.iterrows():
                price = row["Close"]
                if row["positions"] == 1 and cash > price:
                    # buy as many as possible
                    shares = cash // price
                    cash -= shares * price
                elif row["positions"] == -1 and shares > 0:
                    cash += shares * price
                    shares = 0
                nav = cash + shares * price
                equity_curve.append(nav)
            df_bt = pd.DataFrame({"date": df["Date"], "equity": equity_curve})
            st.line_chart(df_bt.set_index("date")["equity"])
            st.success("Backtest (demo) complete. This is illustrative; replace with vectorbt for production-grade testing.")

# --- ALERTS TAB ---
with tabs[5]:
    st.header("Alerts")
    st.markdown("Set price alerts. Alerts are stored in session state and checked whenever you refresh/trigger fetch.")
    a_sym = st.text_input("Ticker for alert", value="", key="alert_sym")
    a_price = st.number_input("Price target", value=0.0, key="alert_price")
    a_type = st.selectbox("Trigger", [">= (cross above)", "<= (cross below)"], key="alert_type")
    if st.button("Create Alert"):
        st.session_state.alerts.append({"symbol": a_sym.upper(), "price": float(a_price), "type": a_type, "created": datetime.utcnow().isoformat(), "notified": False})
        st.success("Alert created.")
    if st.session_state.alerts:
        st.table(pd.DataFrame(st.session_state.alerts))
    # check alerts (simple loop)
    if st.button("Check Alerts Now"):
        triggered = []
        for a in st.session_state.alerts:
            p = fetch_price_yf(a["symbol"]).get("last")
            if p is None: continue
            if a["type"].startswith(">") and p >= a["price"] and not a.get("notified"):
                triggered.append((a, p))
                a["notified"] = True
            if a["type"].startswith("<") and p <= a["price"] and not a.get("notified"):
                triggered.append((a, p))
                a["notified"] = True
        if triggered:
            for t,a_price in triggered:
                st.balloons()
                st.success(f"ALERT: {t['symbol']} {t['type']} {t['price']} — current {a_price}")
        else:
            st.info("No alerts triggered at this time.")

# --- REPORTS TAB ---
with tabs[6]:
    st.header("Reports & Exports")
    st.markdown("Download quick portfolio report (markdown + CSV). Useful for sharing or saving snapshots.")
    if st.button("Generate Snapshot Report"):
        now = datetime.utcnow().isoformat()
        positions = st.session_state.positions
        rows = []
        tot = 0.0
        for p in positions:
            cur = fetch_price_yf(p["symbol"]).get("last") or 0.0
            pl = (cur - p["entry"]) * p["qty"] * (1 if p["side"]=="long" else -1)
            tot += pl
            rows.append({"symbol": p["symbol"], "qty": p["qty"], "entry": p["entry"], "current": cur, "pnl": pl})
        df = pd.DataFrame(rows)
        md = f"# Portfolio Snapshot — {now}\n\n"
        md += f"Total unrealized P&L: ₹{tot:.2f}\n\n"
        md += df.to_markdown(index=False)
        b = io.BytesIO()
        b.write(md.encode("utf-8"))
        b.seek(0)
        st.download_button("Download Markdown Report", b, file_name=f"portfolio_snapshot_{datetime.utcnow().date()}.md", mime="text/markdown")
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button("Download CSV", csv_buf.getvalue(), file_name=f"portfolio_snapshot_{datetime.utcnow().date()}.csv", mime="text/csv")
        st.success("Report ready.")

# ---- Auto-refresh / live behavior note ----
if st.session_state.live:
    # Simple approach: tell the user data auto-refreshes (via cache TTL) and encourage manual refresh if needed.
    st.info("Live mode: data cached with short TTL; use interactive buttons (Check Alerts Now / Run Screener / Run Backtest) to trigger fresh fetches. For true streaming, integrate Polygon WebSocket & Redis (see TODOs).")

# ---- Footer: TODOs + quick tips ----
st.markdown("---")
st.markdown("""
**Next steps / TODOs**
- Plug a professional data provider (Polygon / Finnhub) for real-time ticks and options data.  
- Replace yfinance screener with a dedicated fundamentals DB for faster screening of large universes.  
- Add persistent storage (Supabase / Firebase) for user profiles, watchlists, and alerts.  
- For production backtests, integrate `vectorbt` or `backtrader`.  
- Enable OpenAI (add key to Streamlit secrets and `openai` to requirements) for richer AI features.

**Secrets (Streamlit Cloud)**:
Add keys at *Settings → Secrets*:
