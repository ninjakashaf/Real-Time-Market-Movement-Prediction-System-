"""
Build Time-Series Dataset
==========================
Project: Real-Time Market Movement Prediction System
Phase 3: Merge sentiment + price data into one dataset for deep learning

Run AFTER sentiment_analysis.py
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH  = "market_data.db"
OUT_FILE = "dataset.csv"

# ─── Step 1: Load data from database ──────────────────────────────────────────

def load_data():
    conn = sqlite3.connect(DB_PATH)

    prices = pd.read_sql("""
        SELECT ticker, date, open, high, low, close, volume
        FROM prices
        ORDER BY ticker, date
    """, conn)

    news = pd.read_sql("""
        SELECT published, sentiment_label, sentiment_score
        FROM news
        WHERE sentiment_label IS NOT NULL
    """, conn)

    conn.close()
    print(f"Loaded {len(prices)} price rows and {len(news)} labelled articles.")
    return prices, news


# ─── Step 2: Aggregate sentiment by date ──────────────────────────────────────

def aggregate_sentiment(news):
    # Extract just the date part from published timestamp
    news["date"] = pd.to_datetime(news["published"]).dt.strftime("%Y-%m-%d")

    daily = news.groupby("date").agg(
        avg_sentiment    = ("sentiment_score", "mean"),
        positive_count   = ("sentiment_label", lambda x: (x == "Positive").sum()),
        negative_count   = ("sentiment_label", lambda x: (x == "Negative").sum()),
        neutral_count    = ("sentiment_label", lambda x: (x == "Neutral").sum()),
        total_articles   = ("sentiment_label", "count"),
    ).reset_index()

    daily["sentiment_ratio"] = (
        (daily["positive_count"] - daily["negative_count"]) /
        daily["total_articles"]
    )

    print(f"Aggregated sentiment into {len(daily)} daily rows.")
    return daily


# ─── Step 3: Build features for each ticker ───────────────────────────────────

def build_features(prices, daily_sentiment):
    all_frames = []

    for ticker in prices["ticker"].unique():
        df = prices[prices["ticker"] == ticker].copy()
        df = df.sort_values("date").reset_index(drop=True)

        # Price features
        df["price_change_pct"] = df["close"].pct_change() * 100
        df["price_range"]      = df["high"] - df["low"]
        df["volatility"]       = df["price_change_pct"].rolling(5).std()

        # Direction label: 1 = price went UP next day, 0 = went DOWN
        df["direction"] = (df["close"].shift(-1) > df["close"]).astype(int)

        # Volatility spike label: 1 if next day volatility is high
        df["volatility_spike"] = (
            df["volatility"].shift(-1) > df["volatility"].mean()
        ).astype(int)

        # Merge with sentiment
        df = df.merge(daily_sentiment, on="date", how="left")

        # Fill missing sentiment days with neutral values
        df["avg_sentiment"]  = df["avg_sentiment"].fillna(0)
        df["positive_count"] = df["positive_count"].fillna(0)
        df["negative_count"] = df["negative_count"].fillna(0)
        df["neutral_count"]  = df["neutral_count"].fillna(0)
        df["total_articles"] = df["total_articles"].fillna(0)
        df["sentiment_ratio"]= df["sentiment_ratio"].fillna(0)

        df["ticker"] = ticker
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)

    # Drop rows where we can't calculate direction (last row per ticker)
    combined = combined.dropna(subset=["direction", "price_change_pct"])

    return combined


# ─── Step 4: Create sliding window sequences (for RNN/LSTM/GRU) ───────────────

def create_sequences(df, window=5):
    """
    For each ticker, create overlapping windows of 'window' days.
    Each window becomes one training sample.
    X = features for last N days
    y = direction on the NEXT day
    """
    feature_cols = [
        "close", "volume", "price_change_pct", "price_range",
        "volatility", "avg_sentiment", "positive_count",
        "negative_count", "sentiment_ratio"
    ]

    X_list, y_list, meta = [], [], []

    for ticker in df["ticker"].unique():
        t_df = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        t_df = t_df.dropna(subset=feature_cols)

        for i in range(window, len(t_df)):
            window_data = t_df[feature_cols].iloc[i-window:i].values
            label       = t_df["direction"].iloc[i]
            date        = t_df["date"].iloc[i]

            X_list.append(window_data)
            y_list.append(label)
            meta.append({"ticker": ticker, "date": date})

    X = np.array(X_list)   # shape: (samples, window, features)
    y = np.array(y_list)   # shape: (samples,)

    print(f"\nSequence dataset ready:")
    print(f"  X shape: {X.shape}  (samples x days x features)")
    print(f"  y shape: {y.shape}")
    print(f"  UP days:   {y.sum()} ({round(y.mean()*100)}%)")
    print(f"  DOWN days: {(1-y).sum()} ({round((1-y.mean())*100)}%)")

    return X, y, pd.DataFrame(meta)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_build():
    print("=" * 60)
    print("Building Time-Series Dataset")
    print("=" * 60)

    prices, news = load_data()
    daily_sentiment = aggregate_sentiment(news)
    dataset = build_features(prices, daily_sentiment)

    # Save flat CSV for inspection
    dataset.to_csv(OUT_FILE, index=False)
    print(f"\nFlat dataset saved to: {OUT_FILE}")
    print(f"Total rows: {len(dataset)}")

    # Show a preview
    print("\nSample rows:")
    print("-" * 60)
    preview_cols = ["ticker", "date", "close", "price_change_pct",
                    "avg_sentiment", "positive_count", "negative_count", "direction"]
    print(dataset[preview_cols].head(10).to_string(index=False))

    # Create sequences for deep learning
    print("\n" + "=" * 60)
    print("Creating sequences for RNN / LSTM / GRU...")
    print("=" * 60)
    X, y, meta = create_sequences(dataset, window=5)

    # Save sequences
    np.save("X_sequences.npy", X)
    np.save("y_labels.npy",    y)
    meta.to_csv("meta.csv", index=False)

    print("\nFiles saved:")
    print("  dataset.csv       <- flat data (open in Excel to inspect)")
    print("  X_sequences.npy   <- input for RNN/LSTM/GRU models")
    print("  y_labels.npy      <- labels (0=DOWN, 1=UP)")
    print("  meta.csv          <- dates and tickers for each sequence")
    print("\nPhase 3 COMPLETE! Ready for model training.")

    return X, y, meta


if __name__ == "__main__":
    run_build()
