# app_polygon.py
"""
Bloomberg-Lite — Polygon integration scaffold
- Uses Polygon REST endpoints when POLYGON_API_KEY is present in Streamlit secrets.
- Falls back to yfinance for prototyping.
- Caches responses and includes hooks for WebSocket streaming.
Run: streamlit run app_polygon.py
"""

import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import requests

# optional prototyping fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

st.set_page_config(page_title="Bloomberg-Lite (Polygon)", layout="wide")
st.title("Bloomberg-Lite — Polygon Integration")

# ---------- config ----------
POLY_KEY = st.secrets.get("POLYGON_API_KEY") if st.secrets else None
POLY_BASE = "https://api.polygon.io"

# ---------- helpers: Polygon REST wrappers ----------
def polygon_last_nbbo(ticker: str):
    """
    GET /v2/last/nbbo/{stocksTicker}
    returns latest NBBO (last quote) for ticker.
    Docs: https://polygon.io/docs/rest/stocks/trades-quotes/last-quote
    """
    if not POLY_KEY:
        raise RuntimeError("POLYGON_API_KEY not configured")
    url = f"{POLY_BASE}/v2/last/nbbo/{ticker}"
    params = {"apiKey": POLY_KEY}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def polygon_aggs(ticker: str, multiplier: int, timespan: str, from_date: str, to_date: str, limit: int = 5000):
    """
    GET /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
    Used to fetch OHLC aggregates (bars).
    Docs: https://polygon.io/docs/get_v2_aggs_ticker__ticker__range__multiplier___timespan___from___to
    """
    if not POLY_KEY:
        raise RuntimeError("POLYGON_API_KEY not configured")
    url = f"{POLY_BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {"adjusted": "true", "limit": limit, "apiKey": POLY_KEY}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

# ---------- fallback: yfinance small wrapper ----------
def yfinance_quote_and_history(ticker: str, period="1mo", interval="1d"):
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance not installed; add to requirements or set POLYGON_API_KEY")
    t = yf.Ticker(ticker)
    info = t.info
    hist = t.history(period=period, interval=interval)
    return {"info": info, "history": hist.reset_index()}

# ---------- caching polite wrappers ----------
@st.cache_data(ttl=10)
def get_latest_quote(ticker: str):
    """Return dict with last/prev/pct and original payload."""
    ticker = ticker.upper()
    if POLY_KEY:
        try:
            payload = polygon_last_nbbo(ticker)
            # polygon v2 last nbbo response format
            # we'll extract a few fields -- check payload keys in real responses
            last_quote = payload.get("results", {})
            # last_quote may be empty if not found
            bid = last_quote.get("bid") if isinstance(last_quote, dict) else None
            ask = last_quote.get("ask") if isinstance(last_quote, dict) else None
            last_price = None
            # polygon last NBBO may not include a 'last' field (that's trade); we'll attempt safe fallback
            if isinstance(last_quote, dict):
                # sometimes polygon returns 'last' from trade endpoints; try that too
                last_price = last_quote.get("last", last_quote.get("ask", last_quote.get("bid")))
            return {"source": "polygon", "raw": payload, "bid": bid, "ask": ask, "last": last_price}
        except Exception as e:
            # on any polygon error, fallback to yfinance if available
            st.warning(f"Polygon REST failed: {e} — falling back to yfinance (if available).")
    # fallback:
    if YFINANCE_AVAILABLE:
        y = yfinance_quote_and_history(ticker, period="5d", interval="1d")
        last = None
        try:
            last = y["history"]["Close"].iloc[-1]
        except Exception:
            pass
        return {"source": "yfinance", "raw": y, "last": last}
    return {"error": "No data source available (install yfinance or set POLYGON_API_KEY)."}

@st.cache_data(ttl=300)
def get_ohlcv(ticker: str, multiplier: int = 1, timespan: str = "day", days_back: int = 365):
    """
    Use Polygon aggs for OHLCV if key present, else yfinance.
    returns a pandas-like dict with 'results' or 'history'
    """
    ticker = ticker.upper()
    end = datetime.utcnow().date()
    start = end - timedelta(days=days_back)
    if POLY_KEY:
        try:
            from_date = start.isoformat()
            to_date = end.isoformat()
            payload = polygon_aggs(ticker, multiplier, timespan, from_date, to_date, limit=50000)
            # polygon aggs -> payload['results'] contains bars with o,h,l,c,v,t
            return {"source": "polygon", "raw": payload}
        except Exception as e:
            st.warning(f"Polygon aggs failed: {e} — falling back to yfinance if available.")
    # fallback to yfinance
    if YFINANCE_AVAILABLE:
        y = yfinance_quote_and_history(ticker, period=f"{days_back}d", interval="1d")
        return {"source": "yfinance", "raw": y}
    return {"error": "No data source available."}

# ----------------- UI ------------------
st.sidebar.header("Polygon Settings")
if POLY_KEY:
    st.sidebar.success("POLYGON_API_KEY found in secrets (using Polygon REST).")
else:
    st.sidebar.info("No POLYGON_API_KEY found — using yfinance fallback.")

ticker = st.sidebar.text_input("Ticker", value="AAPL").upper()
if st.sidebar.button("Fetch Latest"):
    q = get_latest_quote(ticker)
    st.sidebar.json(q)

st.header(f"Live Quote — {ticker}")
q_main = get_latest_quote(ticker)
if q_main.get("error"):
    st.error(q_main["error"])
else:
    st.json({"last": q_main.get("last"), "bid": q_main.get("bid"), "ask": q_main.get("ask"), "source": q_main.get("source")})

st.markdown("### OHLCV (last year)")
oh = get_ohlcv(ticker, multiplier=1, timespan="day", days_back=365)
if oh.get("error"):
    st.error(oh["error"])
else:
    if oh.get("source") == "polygon":
        # polygon payload example: payload['results'] = [{o:..., h:..., l:..., c:..., v:..., t:timestamp}]
        res = oh["raw"].get("results", [])
        if not res:
            st.info("Polygon returned no aggregates for this symbol (or your plan doesn't include it).")
        else:
            df = pd.DataFrame(res)
            # polygon timestamp 't' is ms since epoch
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
            st.dataframe(df[['timestamp','Open','High','Low','Close','Volume']].tail(20))
    else:
        # yfinance payload: raw['history'] is a DataFrame-like
        raw = oh['raw']
        try:
            df = raw['history']
            st.dataframe(df.tail(20))
        except Exception as e:
            st.write("Unable to show history:", e)

st.markdown("---")
st.caption("Notes: For streaming WebSocket data, see the short WebSocket example below (run separately).")

# ---------------- WebSocket guidance & sample (NOT run inside Streamlit) ----------------
st.markdown("## WebSocket streaming (example, run outside Streamlit)")
st.code("""
# Example using 'websocket-client' or polygon's official client.
# This snippet is for demonstration — run it as a separate Python process/service.

import websocket, json
API_KEY = "<YOUR_POLYGON_KEY>"
def on_open(ws):
    # authenticate and subscribe to trades
    ws.send(json.dumps({"action":"auth","params":API_KEY}))
    ws.send(json.dumps({"action":"subscribe","params":"T.AAPL"}))  # trades feed star: T.*

def on_message(ws, message):
    data = json.loads(message)
    print("MSG:", data)

ws = websocket.WebSocketApp("wss://socket.polygon.io/stocks", on_open=on_open, on_message=on_message)
ws.run_forever()
""", language="python")

st.markdown("""
**WebSocket notes:** Polygon's Stocks WebSocket feed (trades/quotes/aggregates) is the way to get tick-level live updates. See Polygon docs for feed types and pricing. If you run a WS listener, publish ticks into Redis / PubSub and let Streamlit poll/cache them for display (avoids blocking Streamlit).
""")

# ---------- footer: links & troubleshooting ----------
st.markdown("---")
st.markdown(
    """
**Docs & troubleshooting**
- Polygon Stocks WebSocket & Streams docs (overview + trades/quotes): official docs.  
- Polygon REST endpoints: last NBBO and aggs (used above).  
If you see 403/401 errors, confirm your key and plan permissions in Polygon dashboard.
"""
)

