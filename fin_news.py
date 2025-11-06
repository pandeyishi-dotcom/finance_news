# app.py
"""
Bloomberg-Lite — Enhanced (API-free)
Adds topic clustering, visualization, topic drill-down, improved sentiment, and optional SMTP alerts.
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
from datetime import datetime
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import altair as alt

# ---------------- Config ----------------
DB_PATH = "bloomberg_lite.db"
POLL_INTERVAL_DEFAULT = 15
RSS_QUERY_DEFAULT = "finance OR markets OR stocks"
RSS_MAX_ITEMS = 30
DEFAULT_N_CLUSTERS = 4
TFIDF_MAX_FEATURES = 2000

# ---------------- Helpers / DB ----------------
st.set_page_config(page_title="Bloomberg-Lite — Enhanced", layout="wide")
@st.cache_resource
def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn

def init_db(conn):
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS news_cache (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, link TEXT, summary TEXT, published TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist (symbol TEXT PRIMARY KEY, added_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS prices (symbol TEXT PRIMARY KEY, price REAL, ts TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS positions (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, qty INTEGER, side TEXT, entry REAL, created_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, price REAL, direction TEXT, notified INTEGER DEFAULT 0, created_at TEXT)""")
    conn.commit()

def db_cache_news(conn, items):
    c = conn.cursor()
    c.execute("DELETE FROM news_cache")
    for n in items:
        c.execute("INSERT INTO news_cache(title, link, summary, published) VALUES (?, ?, ?, ?)",
                  (n.get("title"), n.get("link"), n.get("summary"), n.get("published")))
    conn.commit()

def db_get_news(conn):
    c = conn.cursor()
    c.execute("SELECT id, title, link, summary, published FROM news_cache ORDER BY id DESC")
    rows = c.fetchall()
    return [dict(r) for r in rows]

def db_update_price(conn, symbol, price):
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO prices(symbol, price, ts) VALUES (?, ?, ?)", (symbol.upper(), float(price), datetime.utcnow().isoformat()))
    conn.commit()

def db_get_prices(conn):
    c = conn.cursor()
    c.execute("SELECT symbol, price, ts FROM prices")
    return {r["symbol"]: {"price": r["price"], "ts": r["ts"]} for r in c.fetchall()}

def db_add_watch(conn, symbol):
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO watchlist(symbol, added_at) VALUES (?, ?)", (symbol.upper(), datetime.utcnow().isoformat()))
    conn.commit()

def db_get_watch(conn):
    c = conn.cursor()
    c.execute("SELECT symbol FROM watchlist ORDER BY symbol")
    return [r["symbol"] for r in c.fetchall()]

def db_delete_watch(conn, symbol):
    c = conn.cursor()
    c.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol.upper(),))
    conn.commit()

def db_add_position(conn, symbol, qty, side, entry):
    c = conn.cursor()
    c.execute("INSERT INTO positions(symbol, qty, side, entry, created_at) VALUES (?, ?, ?, ?, ?)",
              (symbol.upper(), int(qty), side, float(entry), datetime.utcnow().isoformat()))
    conn.commit()

def db_get_positions(conn):
    c = conn.cursor()
    c.execute("SELECT * FROM positions ORDER BY created_at DESC")
    return [dict(r) for r in c.fetchall()]

def db_add_alert(conn, symbol, price, direction):
    c = conn.cursor()
    c.execute("INSERT INTO alerts(symbol, price, direction, notified, created_at) VALUES (?, ?, ?, 0, ?)",
              (symbol.upper(), float(price), direction, datetime.utcnow().isoformat()))
    conn.commit()

def db_get_alerts(conn):
    c = conn.cursor()
    c.execute("SELECT * FROM alerts ORDER BY created_at DESC")
    return [dict(r) for r in c.fetchall()]

def db_mark_alert_notified(conn, alert_id):
    c = conn.cursor()
    c.execute("UPDATE alerts SET notified = 1 WHERE id = ?", (alert_id,))
    conn.commit()

# ---------------- News fetch ----------------
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

# ---------------- Price fetcher ----------------
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

# ---------------- Poller ----------------
class Poller(threading.Thread):
    def __init__(self, conn, interval=15):
        super().__init__(daemon=True)
        self.conn = conn
        self.interval = max(5, int(interval))
        self._stop = threading.Event()
    def run(self):
        while not self._stop.is_set():
            syms = db_get_watch(self.conn)
            for s in syms:
                try:
                    p = fetch_price_yf(s)
                    if p is not None:
                        db_update_price(self.conn, s, p)
                except Exception:
                    pass
            for _ in range(self.interval):
                if self._stop.is_set(): break
                time.sleep(1)
    def stop(self):
        self._stop.set()

# ---------------- Text processing & clustering ----------------
def build_corpus_from_rows(rows):
    corpus = []
    titles = []
    for r in rows:
        t = (r.get("title") or "") + ". " + (r.get("summary") or "")
        corpus.append(t.strip())
        titles.append(r.get("title") or "")
    return corpus, titles

def tfidf_vectorize(corpus, max_features=TFIDF_MAX_FEATURES):
    vect = TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=(1,2))
    X = vect.fit_transform(corpus)
    return X, vect

def cluster_and_label(X, vect, n_clusters=DEFAULT_N_CLUSTERS):
    # KMeans clustering on TF-IDF
    if X.shape[0] == 0:
        return [], []
    kmeans = KMeans(n_clusters=max(1, int(min(n_clusters, X.shape[0]))), random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    # get top terms per cluster
    terms = np.array(vect.get_feature_names_out())
    centers = kmeans.cluster_centers_
    top_terms = []
    for i in range(centers.shape[0]):
        idx = np.argsort(-centers[i])[:8]
        top = " ".join(terms[idx][:4])
        top_terms.append(top)
    return labels, top_terms, kmeans

def pca_2d(X):
    # reduce to 2D for plotting
    if X.shape[0] <= 2:
        return np.hstack([np.zeros((X.shape[0],1)), np.arange(X.shape[0])[:,None]])
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X.toarray() if hasattr(X, "toarray") else X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    return coords

# ---------------- Sentiment heuristic ----------------
UP_WORDS = {"rise","rises","rose","gain","gains","surge","soar","beat","beats","strong","upgrade","bull","bullish","optimis"}
DOWN_WORDS = {"fall","falls","fell","drop","drops","decline","slump","miss","misses","weak","downgrade","bear","bearish","loss"}

def tone_score(texts):
    s = 0
    for t in texts:
        toks = [w.lower().strip(".,;()[]") for w in re.findall(r"\w+", t)]
        for w in toks:
            if w in UP_WORDS: s += 1
            if w in DOWN_WORDS: s -= 1
    if s > 0: return "Bullish"
    if s < 0: return "Bearish"
    return "Neutral"

# ---------------- SMTP send (optional) ----------------
def send_email_via_smtp(to_email, subject, body):
    # This is optional. Provide SMTP_* keys in .streamlit/secrets.toml to enable.
    try:
        smtp_host = st.secrets.get("SMTP_HOST")
        smtp_port = int(st.secrets.get("SMTP_PORT")) if st.secrets.get("SMTP_PORT") else 587
        smtp_user = st.secrets.get("SMTP_USER")
        smtp_pass = st.secrets.get("SMTP_PASS")
        from_email = st.secrets.get("SMTP_FROM") or smtp_user
        if not smtp_host or not smtp_user or not smtp_pass:
            raise RuntimeError("SMTP credentials not set in secrets.")
        import smtplib, ssl
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        msg.set_content(body)
        ctx = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls(context=ctx)
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True, "Sent"
    except Exception as e:
        return False, str(e)

# ---------------- App UI ----------------
conn = get_db_conn()
if "poller" not in st.session_state:
    st.session_state.poller = None

st.title("📡 Bloomberg-Lite — Enhanced (API-free)")

tabs = st.tabs(["Market", "News & Topics", "Portfolio", "Alerts", "Reports", "Settings"])

# ---------- MARKET ----------
with tabs[0]:
    st.header("Market — Watchlist & Live Prices")
    c1, c2, c3 = st.columns([3,1,1])
    with c1:
        new_sym = st.text_input("Add ticker (e.g., AAPL, RELIANCE.NS)", "")
        if st.button("Add"):
            if new_sym.strip():
                db_add_watch(conn, new_sym.strip())
                st.success(f"Added {new_sym.strip().upper()}")
    with c2:
        interval = st.number_input("Poll interval (secs)", min_value=5, value=POLL_INTERVAL_DEFAULT, step=1)
    with c3:
        if st.session_state.poller and st.session_state.poller.is_alive():
            if st.button("Stop Poller"):
                st.session_state.poller.stop()
                st.session_state.poller = None
                st.success("Stopped poller.")
        else:
            if st.button("Start Poller"):
                poller = Poller(conn, interval=interval)
                poller.start()
                st.session_state.poller = poller
                st.success("Started poller.")

    wl = db_get_watch(conn)
    prices = db_get_prices(conn)
    if wl:
        df_rows = []
        for s in wl:
            pinfo = prices.get(s)
            df_rows.append({"symbol": s, "price": pinfo["price"] if pinfo else None, "updated": pinfo["ts"] if pinfo else None})
        st.table(pd.DataFrame(df_rows))
        sel = st.selectbox("Chart symbol", wl)
        if sel:
            hist = yf.Ticker(sel).history(period="3mo", interval="1d")
            if hist is not None and not hist.empty:
                hist = hist.reset_index()
                st.line_chart(hist.set_index("Date")["Close"])
            else:
                st.info("No historical data.")
    else:
        st.info("Watchlist empty. Add symbols above.")

# ---------- NEWS & TOPICS ----------
with tabs[1]:
    st.header("News & Topic Clustering (local TF-IDF)")
    q, kcol = st.columns([4,1])
    with q:
        query = st.text_input("RSS search query", value=RSS_QUERY_DEFAULT)
    with kcol:
        if st.button("Fetch & Cache News"):
            items = fetch_google_rss(query, max_items=RSS_MAX_ITEMS)
            db_cache_news(conn, items)
            st.success(f"Cached {len(items)} items.")
    news_rows = db_get_news(conn)
    if not news_rows:
        st.info("No news cached. Click 'Fetch & Cache News'.")
    else:
        st.write(f"Cached news: {len(news_rows)} items")
        ncol1, ncol2 = st.columns([2,1])
        with ncol2:
            n_clusters = st.number_input("Number of clusters", min_value=1, max_value=12, value=DEFAULT_N_CLUSTERS, step=1)
            if st.button("Run clustering"):
                with st.spinner("Computing TF-IDF & clusters..."):
                    corpus, titles = build_corpus_from_rows(news_rows)
                    X, vect = tfidf_vectorize(corpus)
                    labels, top_terms, kmeans = cluster_and_label(X, vect, n_clusters=n_clusters)
                    coords = pca_2d(X)
                    # build dataframe for visualization
                    viz_df = pd.DataFrame({
                        "id": [r["id"] for r in news_rows],
                        "title": titles,
                        "cluster": labels,
                        "x": coords[:,0] if coords.shape[0] == len(labels) else np.zeros(len(labels)),
                        "y": coords[:,1] if coords.shape[0] == len(labels) else np.arange(len(labels))
                    })
                    # attach cluster labels
                    cluster_meta = {i: top_terms[i] for i in range(len(top_terms))}
                    st.session_state.viz_df = viz_df
                    st.session_state.cluster_meta = cluster_meta
                    st.success("Clustering complete.")
        # show visualization if present
        if st.session_state.get("viz_df") is not None:
            viz_df = st.session_state.viz_df
            cluster_meta = st.session_state.cluster_meta
            st.subheader("Clusters — click points to inspect")
            chart = alt.Chart(viz_df).mark_circle(size=100).encode(
                x="x:Q",
                y="y:Q",
                color=alt.Color("cluster:N", legend=alt.Legend(title="Cluster")),
                tooltip=["title", "cluster"]
            ).interactive().properties(height=400)
            selected = alt.selection_single(fields=["cluster"], bind="legend", clear=True)
            st.altair_chart(chart.add_selection(selected), use_container_width=True)

            # cluster summary table
            meta_rows = []
            counts = viz_df["cluster"].value_counts().to_dict()
            for cidx, label in cluster_meta.items():
                meta_rows.append({"cluster": cidx, "label_terms": label, "count": int(counts.get(cidx, 0))})
            st.table(pd.DataFrame(meta_rows))

            # drilldown control
            cluster_choice = st.number_input("View cluster #", min_value=int(viz_df["cluster"].min()), max_value=int(viz_df["cluster"].max()), value=int(viz_df["cluster"].min()))
            hits = viz_df[viz_df["cluster"] == cluster_choice]
            st.write(f"Cluster {cluster_choice} — label: {cluster_meta.get(cluster_choice)} — {len(hits)} items")
            # show headlines in cluster
            cluster_ids = set(hits["id"].tolist())
            cluster_rows = [r for r in news_rows if r["id"] in cluster_ids]
            for r in cluster_rows:
                st.markdown(f"**{r['title']}**  \n*{r['published']}*  \n[{r['link']}]({r['link']})")
                if r.get("summary"):
                    st.markdown(f"_{r['summary']}_")
                st.markdown("---")
            # export cluster CSV
            if st.button("Export cluster to CSV"):
                out_df = pd.DataFrame(cluster_rows)
                buf = io.StringIO()
                out_df.to_csv(buf, index=False)
                st.download_button("Download CSV", buf.getvalue(), file_name=f"cluster_{cluster_choice}.csv", mime="text/csv")

            # cluster tone
            cluster_texts = [ (r.get("title") or "") + ". " + (r.get("summary") or "") for r in cluster_rows ]
            if cluster_texts:
                tone = tone_score(cluster_texts)
                st.write(f"Cluster tone (heuristic): **{tone}**")

# ---------- PORTFOLIO ----------
with tabs[2]:
    st.header("Portfolio")
    psym = st.text_input("Symbol to add", "")
    pqty = st.number_input("Qty", min_value=1, value=1)
    pside = st.selectbox("Side", ["long", "short"])
    if st.button("Add position"):
        entry = fetch_price_yf(psym) or 0.0
        db_add_position(conn, psym, pqty, pside, entry)
        st.success(f"Added {psym.upper()} {pqty} {pside} @ {entry:.4f}")
    pos = db_get_positions(conn)
    if pos:
        st.dataframe(pd.DataFrame(pos))
    else:
        st.info("No positions yet.")

# ---------- ALERTS ----------
with tabs[3]:
    st.header("Alerts (with optional SMTP sending)")
    asymbol = st.text_input("Symbol (alert)", key="alert_symbol")
    aprice = st.number_input("Target price", value=0.0, key="alert_price_input")
    adir = st.selectbox("Direction", [">= (cross above)", "<= (cross below)"])
    if st.button("Create alert"):
        if asymbol.strip():
            db_add_alert(conn, asymbol, aprice, adir)
            st.success("Alert created.")
    alerts = db_get_alerts(conn)
    if alerts:
        st.table(pd.DataFrame(alerts))
    else:
        st.info("No alerts configured.")
    if st.button("Run alert check now"):
        prices = db_get_prices(conn)
        any_fire = False
        for a in alerts:
            sym = a["symbol"]
            pinfo = prices.get(sym)
            cur = pinfo["price"] if pinfo else fetch_price_yf(sym)
            if cur is None:
                continue
            if a["direction"].startswith(">=") and cur >= a["price"] and not a["notified"]:
                st.success(f"ALERT: {sym} >= {a['price']} (now {cur})")
                db_mark_alert_notified(conn, a["id"])
                any_fire = True
                # optionally send email if SMTP configured
                if st.secrets.get("SMTP_HOST"):
                    to = st.secrets.get("ALERT_TO_EMAIL")
                    if to:
                        subj = f"Price Alert: {sym} >= {a['price']}"
                        body = f"{sym} crossed above {a['price']} (now {cur})"
                        ok, msg = send_email_via_smtp(to, subj, body)
                        if ok: st.info("Email alert sent.")
                        else: st.warning(f"Email send failed: {msg}")
            if a["direction"].startswith("<=") and cur <= a["price"] and not a["notified"]:
                st.success(f"ALERT: {sym} <= {a['price']} (now {cur})")
                db_mark_alert_notified(conn, a["id"])
                any_fire = True
                if st.secrets.get("SMTP_HOST"):
                    to = st.secrets.get("ALERT_TO_EMAIL")
                    if to:
                        subj = f"Price Alert: {sym} <= {a['price']}"
                        body = f"{sym} crossed below {a['price']} (now {cur})"
                        ok, msg = send_email_via_smtp(to, subj, body)
                        if ok: st.info("Email alert sent.")
                        else: st.warning(f"Email send failed: {msg}")
        if not any_fire:
            st.info("No alerts triggered.")

# ---------- REPORTS ----------
with tabs[4]:
    st.header("Reports & Export")
    if st.button("Export all cached news (CSV)"):
        news_rows = db_get_news(conn)
        if not news_rows:
            st.warning("No news cached.")
        else:
            df = pd.DataFrame(news_rows)
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button("Download news CSV", buf.getvalue(), file_name="news_cache.csv", mime="text/csv")
    if st.button("Export prices"):
        prices = db_get_prices(conn)
        if not prices:
            st.warning("No prices stored.")
        else:
            df = pd.DataFrame([{"symbol":k,"price":v["price"],"ts":v["ts"]} for k,v in prices.items()])
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button("Download prices CSV", buf.getvalue(), file_name="prices.csv", mime="text/csv")

# ---------- SETTINGS ----------
with tabs[5]:
    st.header("Settings & Maintenance")
    st.write("DB path:", DB_PATH)
    if st.button("Clear news cache"):
        c = conn.cursor(); c.execute("DELETE FROM news_cache"); conn.commit(); st.success("Cleared news.")
    if st.button("Reset all data (danger)"):
        with st.expander("Confirm reset"):
            if st.button("Confirm full reset"):
                c = conn.cursor()
                for t in ["news_cache","watchlist","prices","positions","alerts"]:
                    c.execute(f"DELETE FROM {t}")
                conn.commit()
                st.success("Database reset. Please refresh.")
                st.experimental_rerun()
    st.markdown("""
    **SMTP alerts (optional)** — to enable, add these keys to `.streamlit/secrets.toml` or Streamlit Cloud secrets:
    ```
    SMTP_HOST = "smtp.example.com"
    SMTP_PORT = "587"
    SMTP_USER = "your_smtp_user"
    SMTP_PASS = "your_smtp_password"
    SMTP_FROM = "alerts@you.com"  # optional
    ALERT_TO_EMAIL = "you@youremail.com"
    ```
    """)

st.markdown("---")
st.markdown("**Notes:** everything is local & API-free (except yfinance for price lookup). For production you can migrate the SQLite DB to Supabase/Postgres and run the poller as a separate worker.")
