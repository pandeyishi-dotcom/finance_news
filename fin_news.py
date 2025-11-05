import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import io

# ---------- PAGE SETUP ----------
st.set_page_config(
    page_title="Bloomberg-Lite Final",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- THEME ----------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

DARK = """
<style>
body {background-color:#0a0b0c; color:#e6eef3;}
.stMarkdown {color:#e6eef3;}
</style>
"""
LIGHT = """
<style>
body {background-color:#f8fafc; color:#0a0b0c;}
.stMarkdown {color:#0a0b0c;}
</style>
"""

st.session_state.dark_mode = st.sidebar.checkbox(
    "Dark Mode", value=st.session_state.dark_mode
)
st.markdown(DARK if st.session_state.dark_mode else LIGHT, unsafe_allow_html=True)

# ---------- STATE DEFAULTS ----------
defaults = dict(
    watchlist=["AAPL", "MSFT", "TSLA", "RELIANCE.NS"],
    positions=[],
    alerts=[],
    live=False,
)
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ---------- HELPERS ----------
@st.cache_data(ttl=30)
def get_price(sym):
    try:
        t = yf.Ticker(sym)
        hist = t.history(period="5d")
        last = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2] if len(hist) > 1 else last
        info = t.info
        return {"last": float(last), "prev": float(prev), "info": info}
    except Exception:
        return {"last": None, "prev": None, "info": {}}

@st.cache_data(ttl=300)
def get_history(sym, period="1y", interval="1d"):
    try:
        t = yf.Ticker(sym)
        h = t.history(period=period, interval=interval)
        return h.reset_index()
    except Exception:
        return pd.DataFrame()

def pct_color(p):
    if p > 0: return f"<span style='color:lime'>{p:+.2f}%</span>"
    if p < 0: return f"<span style='color:red'>{p:+.2f}%</span>"
    return f"{p:+.2f}%"

# ---------- HEADER ----------
c1, c2 = st.columns([3,1])
with c1:
    st.title("📊 Bloomberg-Lite Final Dashboard")
with c2:
    st.session_state.live = st.checkbox("Live Tape", value=st.session_state.live)

# ---------- LIVE TAPE ----------
def render_tape():
    html = []
    for s in st.session_state.watchlist:
        p = get_price(s)
        if not p["last"]:
            html.append(f"<b>{s}</b> –")
            continue
        change = (p["last"] - p["prev"]) / p["prev"] * 100 if p["prev"] else 0
        html.append(f"<b>{s}</b> {p['last']:.2f} {pct_color(change)}")
    tape = " | ".join(html)
    st.markdown(f"<div style='padding:6px;background:#111827;border-radius:6px'>{tape}</div>", unsafe_allow_html=True)

if st.session_state.live:
    render_tape()

# ---------- TABS ----------
tabs = st.tabs(["Market","News & AI","Portfolio","Screener","Backtest","Alerts","Reports"])

# ===== MARKET =====
with tabs[0]:
    st.header("Market Overview")
    wl = st.text_area("Watchlist (comma separated)", ",".join(st.session_state.watchlist))
    if st.button("Save Watchlist"):
        st.session_state.watchlist = [x.strip().upper() for x in wl.split(",") if x.strip()]
        st.success("Watchlist updated.")
    cols = st.columns(min(4, max(1, len(st.session_state.watchlist))))
    for i, s in enumerate(st.session_state.watchlist):
        p = get_price(s)
        if p["last"] is None: 
            cols[i%4].metric(s, "-", "-")
            continue
        delta = (p["last"] - p["prev"]) / p["prev"] * 100 if p["prev"] else 0
        cols[i%4].metric(s, f"{p['last']:.2f}", f"{delta:+.2f}%")
    st.markdown("---")
    sel = st.selectbox("Select ticker", st.session_state.watchlist)
    h = get_history(sel)
    if not h.empty:
        st.line_chart(h.set_index("Date")["Close"])
    else:
        st.info("No data available.")

# ===== NEWS & AI =====
with tabs[1]:
    st.header("News & AI Insights")
    sym = st.selectbox("Ticker", st.session_state.watchlist, key="news_sym")
    try:
        news_items = yf.Ticker(sym).news[:10]
    except Exception:
        news_items = []
    if news_items:
        for n in news_items:
            st.markdown(f"**{n.get('title','')}**  \n*{n.get('publisher','')}*  \n[{n.get('link','')}]( {n.get('link','')} )")
            st.markdown("---")
    else:
        st.info("No recent news found via yfinance.")

    if "openai" in locals() and "OPENAI_API_KEY" in st.secrets:
        if st.button("Generate AI Summary"):
            import openai
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            titles = "\n".join([n.get("title","") for n in news_items]) or f"No headlines for {sym}"
            prompt = f"Summarize these headlines about {sym} into 3 concise bullets focused on market impact:\n{titles}"
            try:
                r = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=150, temperature=0.3)
                st.subheader("🧠 AI Summary")
                st.write(r.choices[0].text.strip())
            except Exception as e:
                st.error(e)
    else:
        st.caption("Add OPENAI_API_KEY in Secrets to enable AI summaries.")

# ===== PORTFOLIO =====
with tabs[2]:
    st.header("Portfolio Simulator")
    c1,c2,c3 = st.columns(3)
    sym = c1.text_input("Symbol")
    qty = c2.number_input("Qty", min_value=1, value=1)
    side = c3.selectbox("Side", ["long","short"])
    if st.button("Add Position"):
        price = get_price(sym)["last"] or 0
        st.session_state.positions.append({"symbol":sym.upper(),"qty":qty,"side":side,"entry":price})
    if st.session_state.positions:
        df = pd.DataFrame(st.session_state.positions)
        rows, total = [], 0
        for r in df.to_dict("records"):
            cur = get_price(r["symbol"])["last"] or 0
            pnl = (cur - r["entry"])*r["qty"]*(1 if r["side"]=="long" else -1)
            rows.append({**r,"current":cur,"PnL":pnl})
            total += pnl
        st.metric("Total Unrealized P&L", f"₹{total:,.2f}")
        st.dataframe(pd.DataFrame(rows))

# ===== SCREENER =====
with tabs[3]:
    st.header("Screener")
    pe_max = st.slider("Max P/E",5,200,50)
    mc_min = st.number_input("Min MarketCap",0,1_000_000_000_000,0)
    matches=[]
    for s in st.session_state.watchlist:
        info = get_price(s)["info"] or {}
        pe = info.get("trailingPE") or 9999
        mc = info.get("marketCap") or 0
        if pe<=pe_max and mc>=mc_min:
            matches.append({"Symbol":s,"P/E":pe,"MarketCap":mc})
    st.dataframe(pd.DataFrame(matches) if matches else pd.DataFrame())

# ===== BACKTEST =====
with tabs[4]:
    st.header("Backtest – Moving Average Crossover (demo)")
    sym = st.selectbox("Symbol", st.session_state.watchlist, key="bt_sym")
    short = st.number_input("Short MA",5,50,20)
    long = st.number_input("Long MA",10,200,50)
    h = get_history(sym)
    if not h.empty:
        h["SMA_short"]=h["Close"].rolling(short).mean()
        h["SMA_long"]=h["Close"].rolling(long).mean()
        st.line_chart(h.set_index("Date")[["Close","SMA_short","SMA_long"]])
        st.caption("Buy when short MA > long MA; Sell when below.")
    else:
        st.warning("No data available for backtest.")

# ===== ALERTS =====
with tabs[5]:
    st.header("Alerts")
    sym = st.text_input("Symbol for alert", key="alert_sym")
    price = st.number_input("Target price", 0.0, key="alert_price")
    direction = st.selectbox("Direction", [">= (cross above)", "<= (cross below)"], key="alert_dir")
    if st.button("Add Alert"):
        st.session_state.alerts.append({"symbol":sym.upper(),"price":price,"dir":direction,"notified":False})
        st.success("Alert added.")
    st.dataframe(pd.DataFrame(st.session_state.alerts))
    if st.button("Check Alerts Now"):
        for a in st.session_state.alerts:
            cur = get_price(a["symbol"])["last"]
            if cur is None: continue
            trig = (a["dir"].startswith(">=") and cur>=a["price"]) or (a["dir"].startswith("<=") and cur<=a["price"])
            if trig and not a["notified"]:
                st.success(f"🔔 {a['symbol']} {a['dir']} {a['price']} (now {cur})")
                a["notified"]=True

# ===== REPORTS =====
with tabs[6]:
    st.header("Reports & Exports")
    if st.button("Generate Portfolio Report"):
        df = pd.DataFrame(st.session_state.positions)
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        md = f"# Portfolio Report — {now}\n\n" + df.to_markdown(index=False)
        st.download_button("Download Markdown", md, file_name="portfolio.md")
        csv = io.StringIO(); df.to_csv(csv,index=False)
        st.download_button("Download CSV", csv.getvalue(), file_name="portfolio.csv")
# ---------- FOOTER ----------
st.markdown("---")
### ✅ Next Steps
- Integrate Polygon / Finnhub for live data  
- Add persistent storage (Supabase / Firebase)  
- Integrate vectorbt or backtrader for pro backtests  
- Enable OpenAI in Secrets for full AI functionality  
**Secrets Example (Streamlit Cloud):**



