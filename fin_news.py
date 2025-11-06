# bloomberg_lite_ui_v2.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
import sqlite3
import threading
import time
import io
import re
from datetime import datetime
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ----------------------- CONFIG -----------------------
DB_PATH = "bloomberg_lite_ui.db"
POLL_INTERVAL_DEFAULT = 15
RSS_QUERY_DEFAULT = "finance OR markets OR stocks"
RSS_MAX_ITEMS = 30
TFIDF_MAX_FEATURES = 1500
DEFAULT_N_CLUSTERS = 3

# ----------------------- APP SETUP -----------------------
st.set_page_config(page_title="Bloomberg-Lite — UI v2", layout="wide", initial_sidebar_state="expanded")

# --------- database helpers (lightweight sqlite) ---------
@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn

def _init_db(conn):
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist(symbol TEXT PRIMARY KEY, added_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS prices(symbol TEXT PRIMARY KEY, price REAL, ts TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS news_cache(id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, link TEXT, summary TEXT, published TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS positions(id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, qty INTEGER, side TEXT, entry REAL, created_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, price REAL, direction TEXT, notified INTEGER DEFAULT 0, created_at TEXT)""")
    conn.commit()

# convenience DB functions
def db_add_watch(conn, sym):
    if not sym: return
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO watchlist(symbol, added_at) VALUES (?, ?)", (sym.upper().strip(), datetime.utcnow().isoformat()))
    conn.commit()

def db_get_watch(conn):
    c = conn.cursor()
    c.execute("SELECT symbol FROM watchlist ORDER BY symbol")
    return [r['symbol'] for r in c.fetchall()]

def db_remove_watch(conn, sym):
    c = conn.cursor()
    c.execute("DELETE FROM watchlist WHERE symbol = ?", (sym.upper().strip(),))
    conn.commit()

def db_update_price(conn, sym, price):
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO prices(symbol, price, ts) VALUES (?, ?, ?)", (sym.upper().strip(), float(price), datetime.utcnow().isoformat()))
    conn.commit()

def db_get_prices(conn):
    c = conn.cursor()
    c.execute("SELECT symbol, price, ts FROM prices")
    return {r['symbol']:{'price':r['price'],'ts':r['ts']} for r in c.fetchall()}

def db_cache_news(conn, items):
    c = conn.cursor()
    c.execute("DELETE FROM news_cache")
    for n in items:
        c.execute("INSERT INTO news_cache(title, link, summary, published) VALUES (?, ?, ?, ?)", (n.get('title'), n.get('link'), n.get('summary'), n.get('published')))
    conn.commit()

def db_get_news(conn, limit=100):
    c = conn.cursor()
    c.execute("SELECT id, title, link, summary, published FROM news_cache ORDER BY id DESC LIMIT ?", (limit,))
    return [dict(r) for r in c.fetchall()]

def db_add_position(conn, sym, qty, side, entry):
    c = conn.cursor()
    c.execute("INSERT INTO positions(symbol, qty, side, entry, created_at) VALUES (?, ?, ?, ?, ?)", (sym.upper(), int(qty), side, float(entry), datetime.utcnow().isoformat()))
    conn.commit()

def db_get_positions(conn):
    c = conn.cursor()
    c.execute("SELECT * FROM positions ORDER BY created_at DESC")
    return [dict(r) for r in c.fetchall()]

def db_add_alert(conn, sym, price, direction):
    c = conn.cursor()
    c.execute("INSERT INTO alerts(symbol, price, direction, notified, created_at) VALUES (?, ?, ?, 0, ?)", (sym.upper(), float(price), direction, datetime.utcnow().isoformat()))
    conn.commit()

def db_get_alerts(conn):
    c = conn.cursor()
    c.execute("SELECT * FROM alerts ORDER BY created_at DESC")
    return [dict(r) for r in c.fetchall()]

def db_mark_alert(conn, alert_id):
    c = conn.cursor()
    c.execute("UPDATE alerts SET notified=1 WHERE id=?", (int(alert_id),))
    conn.commit()

# ---------------- yfinance helpers ----------------

def fetch_price(sym):
    try:
        t = yf.Ticker(sym)
        df = t.history(period='1d', interval='1m')
        if df is None or df.empty:
            df2 = t.history(period='5d', interval='1d')
            if df2 is None or df2.empty:
                return None
            return float(df2['Close'].iloc[-1])
        return float(df['Close'].iloc[-1])
    except Exception:
        return None

# ---------------- RSS news ----------------
@st.cache_data(ttl=300)
def fetch_news_rss(q=RSS_QUERY_DEFAULT, max_items=30):
    q2 = q.replace(' ', '+')
    url = f"https://news.google.com/rss/search?q={q2}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[:max_items]:
        out.append({'title': getattr(e,'title',''), 'link':getattr(e,'link',''), 'summary': getattr(e,'summary',''), 'published': getattr(e,'published','')})
    return out

# ---------------- Poller thread ----------------
class PricePoller(threading.Thread):
    def __init__(self, conn, interval=POLL_INTERVAL_DEFAULT):
        super().__init__(daemon=True)
        self.conn = conn
        self.interval = max(5,int(interval))
        self._stop = threading.Event()
    def run(self):
        while not self._stop.is_set():
            symbols = db_get_watch(self.conn)
            for s in symbols:
                p = fetch_price(s)
                if p is not None:
                    db_update_price(self.conn, s, p)
            for _ in range(self.interval):
                if self._stop.is_set(): break
                time.sleep(1)
    def stop(self):
        self._stop.set()

# ---------------- Text summarizer & clustering ----------------

def build_corpus(rows):
    corpus = []
    for r in rows:
        corpus.append((r.get('title') or '') + '. ' + (r.get('summary') or ''))
    return corpus

@st.cache_data(ttl=300)
def tfidf_cluster(corpus, n_clusters=DEFAULT_N_CLUSTERS):
    if not corpus:
        return None
    vect = TfidfVectorizer(stop_words='english', max_features=TFIDF_MAX_FEATURES, ngram_range=(1,2))
    X = vect.fit_transform(corpus)
    k = min(int(n_clusters), max(1, X.shape[0]))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    terms = vect.get_feature_names_out()
    centers = km.cluster_centers_
    top_terms = []
    for i in range(centers.shape[0]):
        idx = np.argsort(-centers[i])[:6]
        top_terms.append(' '.join(terms[idx][:4]))
    # reduce to 2D for plotting
    try:
        scaler = StandardScaler(with_mean=False)
        Xs = scaler.fit_transform(X.toarray())
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(Xs)
    except Exception:
        coords = np.zeros((len(corpus),2))
    return dict(labels=labels, top_terms=top_terms, coords=coords)

# ---------------- UI helpers ----------------

def metric_card(symbol, price, prev_price=None, updated=None):
    # returns a small column-block for display
    if price is None:
        st.metric(label=symbol, value='—')
        return
    if prev_price is None:
        change = 0.0
    else:
        change = price - prev_price
    change_pct = (change/prev_price*100) if prev_price and prev_price!=0 else 0.0
    st.metric(label=symbol, value=f"{price:.2f}", delta=f"{change_pct:.2f}%")
    if updated:
        st.caption(f"Updated: {updated}")

# ----------------- App UI -----------------
conn = get_conn()
if 'poller' not in st.session_state:
    st.session_state.poller = None

# Sidebar — controls and quick actions
with st.sidebar:
    st.markdown("# Bloomberg-Lite")
    st.write("**API-free edition — UI v2**")
    st.markdown("---")
    st.subheader("Quick controls")
    watch_sym = st.text_input("Add symbol to watchlist", value="")
    if st.button("Add to watchlist"):
        if watch_sym.strip():
            db_add_watch(conn, watch_sym.strip())
            st.success(f"Added {watch_sym.strip().upper()} to watchlist")
    st.write("")
    poll_interval = st.number_input("Background poll interval (secs)", min_value=5, value=POLL_INTERVAL_DEFAULT)
    if st.session_state.poller and st.session_state.poller.is_alive():
        if st.button("Stop background polling"):
            st.session_state.poller.stop()
            st.session_state.poller = None
            st.success("Stopped poller")
    else:
        if st.button("Start background polling"):
            poller = PricePoller(conn, interval=poll_interval)
            poller.start()
            st.session_state.poller = poller
            st.success("Started background poller")
    st.markdown("---")
    st.subheader("Search & fetch news")
    rss_q = st.text_input("RSS query", value=RSS_QUERY_DEFAULT)
    if st.button("Fetch News (RSS)"):
        items = fetch_news_rss(rss_q, max_items=RSS_MAX_ITEMS)
        db_cache_news(conn, items)
        st.success(f"Fetched {len(items)} items")
    st.markdown("---")
    st.caption("Tip: Use the News & Topics tab to cluster, export and drill into topics.")
    st.markdown("---")
    st.write("Made with ❤️ — local & private")

# Top header with metrics
st.markdown("# Market Dashboard")
col1, col2 = st.columns([3,1])
with col2:
    if st.button("Refresh Prices Now"):
        for s in db_get_watch(conn):
            p = fetch_price(s)
            if p is not None:
                db_update_price(conn, s, p)
        st.success("Prices refreshed")

# show metric cards for top 4 watchlist symbols
watch = db_get_watch(conn)
prices = db_get_prices(conn)
cols = st.columns(4)
for i in range(4):
    with cols[i]:
        if i < len(watch):
            s = watch[i]
            pinfo = prices.get(s)
            if pinfo:
                metric_card(s, pinfo['price'], prev_price=None, updated=pinfo.get('ts'))
            else:
                st.metric(label=s, value='—')
        else:
            st.write("")

st.markdown("---")

# Main tabs
tabs = st.tabs(["Market", "News & Topics", "Portfolio", "Alerts", "Reports", "Help"])

# MARKET tab
with tabs[0]:
    st.subheader("Watchlist & Charts")
    watch = db_get_watch(conn)
    prices = db_get_prices(conn)
    if not watch:
        st.info("Your watchlist is empty — add symbols in the sidebar.")
    else:
        # compact table with actions
        df_rows = []
        for s in watch:
            p = prices.get(s)
            df_rows.append({"symbol": s, "price": (p['price'] if p else None), "updated": (p['ts'] if p else None)})
        st.dataframe(pd.DataFrame(df_rows))
        with st.expander("Chart & details"):
            symbol = st.selectbox("Select symbol to view", watch)
            if symbol:
                hist = yf.Ticker(symbol).history(period='3mo', interval='1d')
                if hist is not None and not hist.empty:
                    hist = hist.reset_index()
                    st.line_chart(hist.set_index('Date')['Close'])
                else:
                    st.info("No historical data available.")

# NEWS & TOPICS
with tabs[1]:
    st.subheader("News — readable cards & topic clustering")
    news_rows = db_get_news(conn)
    if not news_rows:
        st.info("No news cached. Use the sidebar 'Fetch News (RSS)'.")
    else:
        # headline cards
        for r in news_rows[:8]:
            st.markdown(f"### {r['title']}")
            cols = st.columns([5,1])
            cols[0].markdown(f"_{r.get('summary','')[:350]}_")
            cols[1].markdown(f"[{r.get('published','')}]\n\n[Read]({r.get('link','')})")
            st.markdown('---')
        # clustering
        st.markdown("### Topic clustering (TF-IDF + KMeans)")
        k = st.number_input("Number of clusters", min_value=1, max_value=8, value=DEFAULT_N_CLUSTERS)
        if st.button("Run clustering on cached news"):
            corpus = build_corpus(news_rows)
            with st.spinner("Running TF-IDF & clustering..."):
                out = tfidf_cluster(corpus, n_clusters=k)
                if out is None:
                    st.warning("Not enough documents to cluster.")
                else:
                    labels = out['labels']
                    top_terms = out['top_terms']
                    coords = out['coords']
                    # show cluster cards
                    for ci in range(len(top_terms)):
                        col_a, col_b = st.columns([4,1])
                        with col_a:
                            st.write(f"**Topic {ci} — {top_terms[ci]}**")
                            # list headlines in cluster
                            hits = [news_rows[i] for i,l in enumerate(labels) if l==ci]
                            for h in hits[:5]:
                                st.markdown(f"- {h['title']}  ")
                        with col_b:
                            st.write(f"{len(hits)} items")
                    st.success('Clustering done — inspect topic cards above.')

# PORTFOLIO tab
with tabs[2]:
    st.subheader("Portfolio — quick add & export")
    sym = st.text_input("Symbol to add to portfolio", '')
    qty = st.number_input("Qty", min_value=1, value=1)
    side = st.selectbox("Side", ['long','short'])
    if st.button("Add position"):
        entry = fetch_price(sym) or 0.0
        db_add_position(conn, sym, qty, side, entry)
        st.success(f"Added {sym.upper()} {qty} {side} @ {entry:.2f}")
    positions = db_get_positions(conn)
    if positions:
        st.dataframe(pd.DataFrame(positions))
    else:
        st.info('No positions yet.')

# ALERTS tab
with tabs[3]:
    st.subheader("Alerts — create & test")
    a_sym = st.text_input("Alert symbol", key='alert_sym')
    a_price = st.number_input('Target price', value=0.0, format='%.4f')
    a_dir = st.selectbox('Direction', ['>= (cross above)', '<= (cross below)'])
    if st.button('Create Alert'):
        if a_sym.strip():
            db_add_alert(conn, a_sym, a_price, a_dir)
            st.success('Alert created.')
    alerts = db_get_alerts(conn)
    if alerts:
        st.table(pd.DataFrame(alerts))
    else:
        st.info('No alerts configured.')
    if st.button('Run alerts now'):
        prices = db_get_prices(conn)
        fired = False
        for a in alerts:
            sym = a['symbol']
            pinfo = prices.get(sym)
            cur = pinfo['price'] if pinfo else fetch_price(sym)
            if cur is None: continue
            if a['direction'].startswith('>=') and cur >= a['price'] and not a['notified']:
                st.success(f"ALERT: {sym} >= {a['price']} (now {cur})")
                db_mark_alert(conn, a['id'])
                fired = True
            if a['direction'].startswith('<=') and cur <= a['price'] and not a['notified']:
                st.success(f"ALERT: {sym} <= {a['price']} (now {cur})")
                db_mark_alert(conn, a['id'])
                fired = True
        if not fired:
            st.info('No alerts triggered.')

# REPORTS tab
with tabs[4]:
    st.subheader('Reports & Export')
    if st.button('Export news as CSV'):
        rows = db_get_news(conn)
        if not rows:
            st.warning('No news to export')
        else:
            df = pd.DataFrame(rows)
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button('Download news CSV', buf.getvalue(), file_name='news_cache.csv')
    if st.button('Export watchlist'):
        wl = db_get_watch(conn)
        if not wl:
            st.warning('No watchlist symbols')
        else:
            st.download_button('Download watchlist.txt', '\n'.join(wl), file_name='watchlist.txt')

# HELP tab
with tabs[5]:
    st.header('Help & UX tips')
    st.markdown('- Use the **sidebar** to quickly add symbols and fetch news.')
    st.markdown('- Start background polling to keep prices refreshed automatically.')
    st.markdown('- Use the News & Topics tab to cluster headlines into topics and inspect each cluster.')
    st.markdown('- Portfolio and Alerts persist locally in `bloomberg_lite_ui.db` (in project folder).')
    st.markdown('- This UI focuses on readability: larger cards, metrics, and concise actions.')

st.markdown('---')
st.caption('If you want additional visual polish (colors, icons, or custom CSS), I can add a theme + CSS injection next.')
