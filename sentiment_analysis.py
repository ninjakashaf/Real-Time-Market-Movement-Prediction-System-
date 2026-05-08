"""
Sentiment Analysis Pipeline
============================
Project: Real-Time Market Movement Prediction System
Phase 2: Label every news headline as Positive / Negative / Neutral

Requirements:
    pip install vaderSentiment pandas

Run AFTER data_ingestion.py
"""

import sqlite3
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── Config ───────────────────────────────────────────────────────────────────

DB_PATH = "market_data.db"

# ─── Setup ────────────────────────────────────────────────────────────────────

def get_sentiment_label(compound_score):
    """
    Convert VADER compound score to simple label.
    compound is between -1.0 (very negative) and +1.0 (very positive)
    """
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# ─── Main ─────────────────────────────────────────────────────────────────────

def run_sentiment():
    print("=" * 60)
    print("Starting Sentiment Analysis")
    print("=" * 60)

    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    # Add sentiment columns to news table if they don't exist yet
    try:
        conn.execute("ALTER TABLE news ADD COLUMN sentiment_label TEXT")
        conn.execute("ALTER TABLE news ADD COLUMN sentiment_score REAL")
        conn.commit()
        print("Added sentiment columns to database.")
    except sqlite3.OperationalError:
        print("Sentiment columns already exist. Continuing...")

    # Load all news headlines
    df = pd.read_sql("SELECT id, title, summary FROM news", conn)
    print(f"\nFound {len(df)} articles to analyse.\n")

    # Create VADER analyser
    analyser = SentimentIntensityAnalyzer()

    results = []

    for _, row in df.iterrows():
        # Combine title + summary for better accuracy
        text = row["title"]
        if row["summary"]:
            text = text + ". " + str(row["summary"])

        # Get sentiment scores
        scores = analyser.polarity_scores(text)
        compound = scores["compound"]
        label = get_sentiment_label(compound)

        # Save back to database
        conn.execute(
            "UPDATE news SET sentiment_label=?, sentiment_score=? WHERE id=?",
            (label, compound, row["id"])
        )

        results.append({
            "title":   row["title"][:60] + "..." if len(row["title"]) > 60 else row["title"],
            "score":   round(compound, 3),
            "label":   label
        })

    conn.commit()

    # ─── Show Results ─────────────────────────────────────────────────────────

    results_df = pd.DataFrame(results)

    print("SENTIMENT RESULTS:")
    print("-" * 80)
    for _, r in results_df.iterrows():
        emoji = "🟢" if r["label"] == "Positive" else "🔴" if r["label"] == "Negative" else "⚪"
        print(f"{emoji} [{r['score']:+.3f}] {r['title']}")

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    counts = results_df["label"].value_counts()
    total  = len(results_df)
    for label, count in counts.items():
        pct = round(count / total * 100)
        bar = "█" * (pct // 5)
        print(f"  {label:<10} {count:>3} articles  {bar} {pct}%")

    print(f"\n  Total: {total} articles analysed")
    print(f"  Saved to: {DB_PATH}")
    print("=" * 60)

    conn.close()
    return results_df


if __name__ == "__main__":
    run_sentiment()
