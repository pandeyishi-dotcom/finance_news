# app.py
"""
Bloomberg-Lite (Advanced, API-free)
- SQLite for persistence (watchlist, positions, alerts, price history)
- Background polling worker using yfinance
- TF-IDF summarizer over RSS headlines + short summaries (no external LLM)
- Simple sentiment/tone scoring
- Streamlit UI: Market / News / Portfolio / Alerts / Reports
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
import threading
import sqlite3
import time
import io
import re
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text as sk_text
from collections import Counter, defaultdict

# ----------------- CONFIG -----------------
DB_PATH = "bloomberg_lite.db"
POLL_INTERVAL_DEFAULT = 15  # seconds
RSS_QUERY_DEFAULT = "finance OR markets OR stocks"
RSS_MAX_ITEMS = 20
TFIDF_TOP_K = 3  # number of bullets

# ----------------- UTIL -----------------
st.set_page_config(page_title="Bloomberg-Lite — Advanced (API-free)", layout="wide")

@st.cache_resource
def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn

def init_db(conn):
    c = conn.cursor()
    # watchlist table
    c.execute("""
    CREATE TABLE IF NOT EXISTS watchlist (
        symbol TEXT PRIMARY KEY,
        added_at TEXT
    )
    """)
    # positions
    c.execute("""
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        qty INTEGER,
        side TEXT,
        entry REAL,
        created_at TEXT
    )
    """)
    # alerts
    c.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        price REAL,
        direction TEXT,
        notified INTEGER DEFAULT 0,
        created_at TEXT
    )
    """)
    # simple price history store (most recent tick)
    c.execute("""
    CREATE TABLE IF NOT EXISTS prices (
        symbol TEXT PRIMARY KEY,
        price REAL,
        ts TEXT
    )
    """)
    # news cache
    c.execute("""
    CREATE TABLE IF NOT EXISTS news_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        link TEXT,
        summary TEXT,
        published TEXT
    )
    """)
    conn.commit()

def db_insert_watchlist(conn, symbol):
    symbol = symbol.upper().strip()
    if not symbol:
        return
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO watchlist(symbol, added_at) VALUES (?, ?)", (symbol, datetime.utcnow().isoformat()))
    conn.commit()

def db_delete_watchlist(conn, symbol):
    c = conn.cursor()
    c.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol.upper().strip(),))
    conn.commit()

def db_get_watchlist(conn):
    c = conn.cursor()
    c.execute("SELECT symbol FROM watchlist ORDER BY symbol")
    return [r["symbol"] for r in c.fetchall()]

def db_add_position(conn, symbol, qty, side, entry):
    c = conn.cursor()
    c.execute("INSERT INTO positions(symbol, qty, side, entry, created_at) VALUES (?, ?, ?, ?, ?)",
              (symbol.upper(), int(qty), side, float(entry), datetime.utcnow().isoformat()))
    conn.commit()

def db_get_positions(conn):
    c = conn.cursor()
    c.execute("SELECT * FROM positions ORDER BY created_at DESC")
    return pd.DataFrame(c.fetchall()) if c.fetchall() is None else pd.DataFrame([dict(row) for row in c.execute("SELECT * FROM positions ORDER BY created_at DESC")])

def db_get_positions_rows(conn):
    c = conn.cursor()
    c.execute("SELECT * FROM positions ORDER BY created_at DESC")
    rows = c.fetchall()
    return [dict(r) for r in rows]

def db_add_alert(conn, symbol, price, direction):
    c = conn.cursor()
    c.execute("INSERT INTO alerts(symbol, price, direction, notified, created_at) VALUES (?, ?, ?, 0, ?)",
              (symbol.upper(), float(price), direction, datetime.utcnow().isoformat()))
    conn.commit()

def db_get_alerts(conn):
    c = conn.cursor()
    c.execute("SELECT * FROM alerts ORDER BY created_at DESC")
    return [dict(r) for r in c.fetchall()]

def db_update_alert_notified(conn, alert_id):
    c = conn.cursor()
    c.execute("UPDATE alerts SET notified = 1 WHERE id = ?", (alert_id,))
    conn.commit()

def db_update_price(conn, symbol, price):
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO prices(symbol, price, ts) VALUES (?, ?, ?)", (symbol.upper(), float(price), datetime.utcnow().isoformat()))
    conn.commit()

def db_get_prices(conn):
    c = conn.cursor()
    c.execute("SELECT symbol, price, ts FROM prices")
    return {r["symbol"]: {"price": r["price"], "ts": r["ts"]} for r in c.fetchall()}

def db_cache_news(conn, news_items):
    c = conn.cursor()
    # replace cache for simplicity
    c.execute("DELETE FROM news_cache")
    for n in news_items:
        c.execute("INSERT INTO news_cache(title, link, summary, published) VALUES (?, ?, ?, ?)",
                  (n.get("title"), n.get("link"), n.get("summary"), n.get("published")))
    conn.commit()

def db_get_news_cache(conn, limit=50):
    c = conn.cursor()
    c.execute("SELECT title, link, summary, published FROM news_cache ORDER BY id DESC LIMIT ?", (limit,))
    return [dict(r) for r in c.fetchall()]

# ----------------- POLLING WORKER -----------------
class Poller(threading.Thread):
    def __init__(self, conn, interval=POLL_INTERVAL_DEFAULT):
        super().__init__(daemon=True)
        self.conn = conn
        self.interval = interval
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            # fetch watchlist
            syms = db_get_watchlist(self.conn)
            for s in syms:
                try:
                    p = fetch_price_yf(s)
                    if p is not None:
                        db_update_price(self.conn, s, p)
                except Exception as e:
                    # swallow errors - continue
                    pass
            # sleep in small increments to allow quicker stop
            for _ in range(int(max(1, self.interval))):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def stop(self):
        self._stop_event.set()

# ----------------- YFINANCE HELPER -----------------
def fetch_price_yf(symbol):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period="1d", interval="1m")
        if df is None or df.empty:
            df2 = t.history(period="5d", interval="1d")
            if df2 is None or df2.empty:
                return None
            return float(df2["Close"].iloc[-1])
        return float(df["Close"].iloc[-1])
    except Exception:
        return None

# ----------------- NEWS FETCH (Google News RSS) -----------------
def fetch_google_rss(query=RSS_QUERY_DEFAULT, max_items=RSS_MAX_ITEMS):
    q_ = query.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={q_}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:max_items]:
        items.append({
            "title": getattr(e, "title", ""),
            "link": getattr(e, "link", ""),
            "summary": getattr(e, "summary", ""),
            "published": getattr(e, "published", "")
        })
    return items

# ----------------- LOCAL TF-IDF SUMMARIZER -----------------
ENGLISH_STOP_WORDS = sk_text.ENGLISH_STOP_WORDS

def build_corpus_from_news(news_items):
    # join title + summary for richer corpus
    corpus = []
    for n in news_items:
        text = (n.get("title","") or "") + ". " + (n.get("summary","") or "")
        corpus.append(text.strip())
    return corpus

def extract_top_phrases_tfidf(corpus, top_k=TFIDF_TOP_K):
    # Heuristic summarization:
    # - compute TF-IDF over the corpus
    # - score sentences by similarity to top TF-IDF features
    if not corpus:
        return []
    # create sentences: split on punctuation/newline
    sentences = []
    for doc in corpus:
        sents = re.split(r'(?<=[\.\!\?])\s+', doc)
        for s in sents:
            s_clean = s.strip()
            if s_clean:
                sentences.append(s_clean)
    # if sentences too few, return docs themselves truncated
    if not sentences:
        return corpus[:top_k]

    # vectorize sentences
    vect = TfidfVectorizer(stop_words='english', max_features=4000, ngram_range=(1,2))
    try:
        X = vect.fit_transform(sentences)
    except Exception:
        # fallback
        return sentences[:top_k]
    # compute sentence scores by sum of TF-IDF weights (or centrality)
    scores = np.asarray(X.sum(axis=1)).ravel()
    ranked_idx = np.argsort(-scores)
    top_sentences = [sentences[i] for i in ranked_idx[:top_k]]
    # dedupe and shorten
    out = []
    seen = set()
    for s in top_sentences:
        normalized = s.lower().strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        if len(s) > 220:
            s = s[:217] + "…"
        out.append(s)
        if len(out) >= top_k:
            break
    return out

# ----------------- SIMPLE SENTIMENT/TONE -----------------
UP_WORDS = {"rise","rises","rose","jump","jumps","surge","surges","gain","gains","soar","up","beat","beats","beatest","record","bull","bullish","optimis"}
DOWN_WORDS = {"fall","falls","fell","drop","drops","decline","declines","slump","down","slide","tumble","loss","losses","miss","misses","bear","bearish","pessim"}

def tone_score_from_text(texts):
    score = 0
    for t in texts:
        toks = [w.lower().strip(".,;()[]") for w in re.findall(r"\w+", t)]
        for w in toks:
            if w in UP_WORDS:
                score += 1
            if w in DOWN_WORDS:
                score -= 1
    if score > 0:
        return "Bullish"
    if score < 0:
        return "Bearish"
    return "Neutral"

# ----------------- APP STATE & UI -----------------
conn = get_db_conn()

# Poller management (persist poller object in session_state)
if "poller" not in st.session_state:
    st.session_state.poller = None
if "poll_interval" not in st.session_state:
    st.session_state.poll_interval = POLL_INTERVAL_DEFAULT

st.title("📡 Bloomberg-Lite — Advanced (API-free)")

tabs = st.tabs(["Market", "News", "Portfolio", "Alerts", "Reports", "Settings"])

# ---------- MARKET TAB ----------
with tabs[0]:
    st.header("Market — Live Prices & Charts")
    col1, col2, col3 = st.columns([3,1,1])
    with col1:
        new_sym = st.text_input("Add ticker to watchlist (e.g., AAPL, RELIANCE.NS)", "")
        if st.button("Add symbol"):
            if new_sym.strip():
                db_insert_watchlist(conn, new_sym.strip())
                st.success(f"Added {new_sym.strip().upper()} to watchlist.")
    with col2:
        st.session_state.poll_interval = st.number_input("Poll interval (secs)", min_value=5, value=st.session_state.poll_interval, step=1)
    with col3:
        if st.session_state.poller and st.session_state.poller.is_alive():
            if st.button("Stop background polling"):
                st.session_state.poller.stop()
                st.session_state.poller = None
                st.success("Polling stopped.")
        else:
            if st.button("Start background polling"):
                poller = Poller(conn, interval=st.session_state.poll_interval)
                poller.start()
                st.session_state.poller = poller
                st.success("Background polling started.")

    # show watchlist
    wl = db_get_watchlist(conn)
    if not wl:
        st.info("Watchlist empty — add tickers above.")
    else:
        st.write("Watchlist:")
        cols = st.columns([2,1,1,1])
        cols[0].write("Symbol")
        cols[1].write("Last price")
        cols[2].write("Updated")
        cols[3].write("Actions")
        prices = db_get_prices(conn)
        for s in wl:
            pinfo = prices.get(s)
            price_text = f"{pinfo['price']:.4f}" if pinfo and pinfo.get("price") is not None else "--"
            updated = pinfo.get("ts") if pinfo else "--"
            with st.container():
                cols = st.columns([2,1,1,1])
                cols[0].write(s)
                cols[1].write(price_text)
                cols[2].write(updated)
                if cols[3].button(f"Remove##{s}"):
                    db_delete_watchlist(conn, s)
                    st.experimental_rerun()

        # show chart for a selected symbol
        sel = st.selectbox("Select symbol for chart", wl)
        if sel:
            df = yf.Ticker(sel).history(period="3mo", interval="1d")
            if df is not None and not df.empty:
                df = df.reset_index()
                df["Date"] = pd.to_datetime(df["Date"])
                st.line_chart(df.set_index("Date")["Close"])
            else:
                st.info("No historical data available for this ticker.")

# ---------- NEWS TAB ----------
with tabs[1]:
    st.header("News — TF-IDF Summaries (local, API-free)")
    q_col1, q_col2 = st.columns([4,1])
    with q_col1:
        query = st.text_input("Search query for Google News RSS", value=RSS_QUERY_DEFAULT)
    with q_col2:
        if st.button("Fetch latest"):
            news_items = fetch_google_rss(query, max_items=RSS_MAX_ITEMS)
            db_cache_news(conn, news_items)
            st.success(f"Fetched {len(news_items)} items.")

    # load cached news
    cached = db_get_news_cache(conn, limit=RSS_MAX_ITEMS)
    if not cached:
        st.info("No news cached — click Fetch latest to pull from Google News RSS.")
    else:
        # show headlines
        for n in cached[:12]:
            st.markdown(f"**{n['title']}**  \n*{n['published']}*  \n[{n['link']}]({n['link']})")
            if n.get("summary"):
                st.markdown(f"_{n['summary']}_")
            st.markdown("---")

        if st.button("Generate Local TF-IDF Summary"):
            corpus = build_corpus_from_news(cached)
            top_sentences = extract_top_phrases_tfidf(corpus, top_k=TFIDF_TOP_K)
            tone = tone_score_from_text([c for c in corpus])
            st.subheader("Local summary")
            for i, s in enumerate(top_sentences, 1):
                st.write(f"{i}. {s}")
            st.write(f"**Tone (heuristic):** {tone}")
            # also show keywords
            keywords = Counter()
            for c in corpus:
                for tok in re.findall(r"\w{3,}", c.lower()):
                    if tok not in ENGLISH_STOP_WORDS:
                        keywords[tok] += 1
            top_keywords = [k for k,_ in keywords.most_common(8)]
            if top_keywords:
                st.write("**Top keywords:**", ", ".join(top_keywords[:8]))

# ---------- PORTFOLIO TAB ----------
with tabs[2]:
    st.header("Portfolio")
    c1, c2, c3 = st.columns([3,1,1])
    with c1:
        psym = st.text_input("Symbol to add to portfolio", "")
    with c2:
        pqty = st.number_input("Qty", min_value=1, value=1, step=1)
    with c3:
        pside = st.selectbox("Side", ["long", "short"])
    if st.button("Add position"):
        entry = fetch_price_yf(psym) or 0.0
        db_add_position(conn, psym, pqty, pside, float(entry))
        st.success(f"Added {psym.upper()} {pqty} {pside} @ {entry:.4f}")
    # show positions
    pos_rows = db_get_positions_rows(conn)
    if pos_rows:
        st.dataframe(pd.DataFrame(pos_rows))
    else:
        st.info("No positions yet.")

# ---------- ALERTS TAB ----------
with tabs[3]:
    st.header("Alerts")
    a1, a2, a3 = st.columns([2,1,1])
    with a1:
        asymbol = st.text_input("Alert symbol", key="alert_symbol")
    with a2:
        aprice = st.number_input("Target price", value=0.0, key="alert_price")
    with a3:
        adir = st.selectbox("Direction", [">= (cross above)", "<= (cross below)"], key="alert_dir")
    if st.button("Create alert"):
        if asymbol.strip():
            db_add_alert(conn, asymbol, aprice, adir)
            st.success("Alert created.")
    # list alerts
    alerts = db_get_alerts(conn)
    if alerts:
        for a in alerts:
            st.write(f"{a['id']}: {a['symbol']} {a['direction']} {a['price']} — Notified: {bool(a['notified'])}")
    else:
        st.info("No alerts configured.")

    if st.button("Run alert check now"):
        prices = db_get_prices(conn)
        any_alert = False
        for a in alerts:
            sym = a['symbol']
            pinfo = prices.get(sym)
            cur = pinfo['price'] if pinfo else fetch_price_yf(sym)
            if cur is None:
                continue
            if a['direction'].startswith(">=") and cur >= a['price'] and not a['notified']:
                st.success(f"ALERT: {sym} crossed above {a['price']} (now {cur})")
                db_update_alert_notified(conn, a['id'])
                any_alert = True
            if a['direction'].startswith("<=") and cur <= a['price'] and not a['notified']:
                st.success(f"ALERT: {sym} crossed below {a['price']} (now {cur})")
                db_update_alert_notified(conn, a['id'])
                any_alert = True
        if not any_alert:
            st.info("No alerts triggered.")

# ---------- REPORTS TAB ----------
with tabs[4]:
    st.header("Reports & Export")
    if st.button("Export positions to CSV"):
        pos = db_get_positions_rows(conn)
        if not pos:
            st.warning("No positions to export.")
        else:
            df = pd.DataFrame(pos)
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button("Download CSV", buf.getvalue(), file_name="positions.csv", mime="text/csv")

    if st.button("Export watchlist"):
        wl = db_get_watchlist(conn)
        if not wl:
            st.warning("No watchlist symbols.")
        else:
            st.download_button("Download watchlist.txt", "\n".join(wl), file_name="watchlist.txt", mime="text/plain")

# ---------- SETTINGS TAB ----------
with tabs[5]:
    st.header("Settings & Utilities")
    st.write("Database path:", DB_PATH)
    if st.button("Clear news cache"):
        c = conn.cursor()
        c.execute("DELETE FROM news_cache")
        conn.commit()
        st.success("Cleared news cache.")
    if st.button("Reset watchlist & positions (danger)"):
        with st.expander("Confirm reset"):
            st.write("This will delete all watchlist, positions, alerts, prices and news.")
            if st.button("Confirm hard reset"):
                c = conn.cursor()
                c.execute("DELETE FROM watchlist")
                c.execute("DELETE FROM positions")
                c.execute("DELETE FROM alerts")
                c.execute("DELETE FROM prices")
                c.execute("DELETE FROM news_cache")
                conn.commit()
                st.success("Database reset. Please refresh the page.")
                st.experimental_rerun()

# footer
st.markdown("---")
st.markdown("**Notes:** This app uses yfinance (free, may be delayed), Google News RSS (free) and local TF-IDF summarization + simple sentiment heuristics. No external paid APIs or keys required.")
