"""
Data Ingestion Pipeline
=======================
Project: Real-Time Market Movement Prediction System
Sources: Yahoo Finance, Reuters RSS, Reddit (r/stocks, r/investing)

Requirements:
    pip install yfinance praw feedparser pandas requests
"""

import os
import time
import sqlite3
import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import feedparser
import praw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

TICKERS = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]

REDDIT_CLIENT_ID     = "SKIP"
REDDIT_CLIENT_SECRET = "SKIP"
REDDIT_USER_AGENT    = "MarketSentimentBot/1.0"

SUBREDDITS = ["stocks", "investing", "wallstreetbets", "SecurityAnalysis"]
REDDIT_POST_LIMIT = 50

RSS_FEEDS = {
    "reuters_business":  "https://feeds.reuters.com/reuters/businessNews",
    "reuters_markets":   "https://feeds.reuters.com/reuters/companyNews",
    "reuters_tech":      "https://feeds.reuters.com/reuters/technologyNews",
    "yahoo_finance_rss": "https://finance.yahoo.com/news/rssindex",
}

DB_PATH = "market_data.db"
PRICE_HISTORY_DAYS = 90


# ─── Database Setup ───────────────────────────────────────────────────────────

def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker     TEXT NOT NULL,
            date       TEXT NOT NULL,
            open       REAL,
            high       REAL,
            low        REAL,
            close      REAL,
            volume     INTEGER,
            fetched_at TEXT DEFAULT (datetime('now')),
            UNIQUE(ticker, date)
        );
        CREATE TABLE IF NOT EXISTS news (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            source     TEXT NOT NULL,
            title      TEXT NOT NULL,
            summary    TEXT,
            url        TEXT,
            published  TEXT,
            fetched_at TEXT DEFAULT (datetime('now')),
            UNIQUE(url)
        );
        CREATE TABLE IF NOT EXISTS reddit_posts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id      TEXT UNIQUE NOT NULL,
            subreddit    TEXT NOT NULL,
            title        TEXT NOT NULL,
            selftext     TEXT,
            score        INTEGER,
            num_comments INTEGER,
            created_utc  TEXT,
            url          TEXT,
            fetched_at   TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    log.info("Database initialised: %s", db_path)
    return conn


# ─── Yahoo Finance ────────────────────────────────────────────────────────────

def fetch_prices(conn, tickers=TICKERS, days=PRICE_HISTORY_DAYS):
    end   = datetime.today()
    start = end - timedelta(days=days)
    all_frames = []

    for ticker in tickers:
        try:
            log.info("Fetching price data for %s ...", ticker)
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )

            if df.empty:
                log.warning("No data returned for %s", ticker)
                continue

            df = df.reset_index()

            # Fix for newer yfinance versions that return tuple column names
            new_cols = []
            for c in df.columns:
                if isinstance(c, tuple):
                    new_cols.append(c[0].lower())
                else:
                    new_cols.append(c.lower())
            df.columns = new_cols

            df["ticker"] = ticker
            df["date"]   = df["date"].astype(str)

            rows = df[["ticker", "date", "open", "high", "low", "close", "volume"]].values.tolist()
            conn.executemany(
                """INSERT OR IGNORE INTO prices
                   (ticker, date, open, high, low, close, volume)
                   VALUES (?,?,?,?,?,?,?)""",
                rows,
            )
            conn.commit()
            all_frames.append(df)
            log.info("  -> %d rows saved for %s", len(df), ticker)

        except Exception as exc:
            log.error("Error fetching %s: %s", ticker, exc)

    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


# ─── RSS Feeds ────────────────────────────────────────────────────────────────

def fetch_rss(conn, feeds=RSS_FEEDS):
    records = []

    for feed_name, url in feeds.items():
        log.info("Fetching RSS feed: %s ...", feed_name)
        try:
            parsed  = feedparser.parse(url)
            entries = parsed.get("entries", [])
            log.info("  -> %d entries found", len(entries))

            for entry in entries:
                title     = entry.get("title", "").strip()
                summary   = entry.get("summary", "").strip()
                link      = entry.get("link", "").strip()
                published = entry.get("published", "") or entry.get("updated", "")

                if not title or not link:
                    continue

                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO news
                           (source, title, summary, url, published)
                           VALUES (?,?,?,?,?)""",
                        (feed_name, title, summary, link, published),
                    )
                    records.append({
                        "source": feed_name, "title": title,
                        "summary": summary, "url": link, "published": published,
                    })
                except sqlite3.IntegrityError:
                    pass

            conn.commit()

        except Exception as exc:
            log.error("Error parsing feed %s: %s", feed_name, exc)

        time.sleep(1)

    log.info("RSS: %d total articles saved this run", len(records))
    return pd.DataFrame(records)


# ─── Reddit ───────────────────────────────────────────────────────────────────

def fetch_reddit(conn, subreddits=SUBREDDITS, limit=REDDIT_POST_LIMIT):
    if REDDIT_CLIENT_ID in ("SKIP", "YOUR_CLIENT_ID"):
        log.warning("Reddit credentials not set -- skipping Reddit ingestion.")
        return pd.DataFrame()

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        check_for_async=False,
    )

    records = []

    for sub_name in subreddits:
        log.info("Fetching r/%s ...", sub_name)
        try:
            subreddit = reddit.subreddit(sub_name)
            posts = list(subreddit.hot(limit=limit // 2)) + \
                    list(subreddit.new(limit=limit // 2))

            for post in posts:
                created = datetime.utcfromtimestamp(post.created_utc).isoformat()
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO reddit_posts
                           (post_id, subreddit, title, selftext, score,
                            num_comments, created_utc, url)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (post.id, sub_name, post.title, post.selftext[:2000],
                         post.score, post.num_comments, created, post.url),
                    )
                    records.append({
                        "post_id": post.id, "subreddit": sub_name,
                        "title": post.title, "score": post.score,
                        "num_comments": post.num_comments, "created_utc": created,
                    })
                except sqlite3.IntegrityError:
                    pass

            conn.commit()
            log.info("  -> %d posts saved from r/%s", len(records), sub_name)

        except Exception as exc:
            log.error("Error fetching r/%s: %s", sub_name, exc)

        time.sleep(2)

    log.info("Reddit: %d total posts saved this run", len(records))
    return pd.DataFrame(records)


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(conn):
    cur = conn.cursor()
    for table in ("prices", "news", "reddit_posts"):
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        log.info("Table %-15s -> %d rows total", table, count)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_pipeline():
    log.info("=" * 60)
    log.info("Starting data ingestion pipeline")
    log.info("=" * 60)

    conn   = init_db()
    prices = fetch_prices(conn)
    news   = fetch_rss(conn)
    reddit = fetch_reddit(conn)

    print_summary(conn)
    log.info("Pipeline complete. Database saved to: %s", DB_PATH)
    conn.close()

    return {"prices": prices, "news": news, "reddit": reddit}


if __name__ == "__main__":
    data = run_pipeline()

    for name, df in data.items():
        if not df.empty:
            print(f"\n-- {name.upper()} (last 5 rows) --")
            print(df.tail(5).to_string(index=False))