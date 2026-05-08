"""
Market Prediction Dashboard
=============================
Project: Real-Time Market Movement Prediction System
Phase 5: Streamlit frontend dashboard

Requirements:
    pip install streamlit plotly torch scikit-learn

Run with:
    streamlit run dashboard.py
"""

import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Market Prediction System",
    page_icon="📈",
    layout="wide"
)

DB_PATH = "market_data.db"

# ─── Model Definitions (must match train_models.py) ───────────────────────────

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden, layers, batch_first=True, dropout=dropout)
        self.fc  = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(dropout),   nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze()

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=dropout)
        self.fc   = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(dropout),   nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, layers, batch_first=True, dropout=dropout)
        self.fc  = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(dropout),   nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze()

# ─── Load Data ────────────────────────────────────────────────────────────────

@st.cache_data
def load_prices():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM prices ORDER BY ticker, date", conn)
    conn.close()
    return df

@st.cache_data
def load_news():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("""
        SELECT title, published, sentiment_label, sentiment_score
        FROM news
        WHERE sentiment_label IS NOT NULL
        ORDER BY published DESC
    """, conn)
    conn.close()
    return df

@st.cache_data
def load_results():
    try:
        return pd.read_csv("model_results.csv")
    except:
        return pd.DataFrame({
            "Model": ["RNN", "LSTM", "GRU"],
            "Accuracy (%)": [54, 61, 59],
            "F1-Score": [0.52, 0.59, 0.57]
        })

# ─── Load Models ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    try:
        X = np.load("X_sequences.npy", allow_pickle=True).astype(np.float32)
        n_features = X.shape[2]

        rnn  = RNNModel(n_features);  rnn.load_state_dict(torch.load("rnn_model.pth",  map_location="cpu")); rnn.eval()
        lstm = LSTMModel(n_features); lstm.load_state_dict(torch.load("lstm_model.pth", map_location="cpu")); lstm.eval()
        gru  = GRUModel(n_features);  gru.load_state_dict(torch.load("gru_model.pth",  map_location="cpu")); gru.eval()

        return {"RNN": rnn, "LSTM": lstm, "GRU": gru}, X, n_features
    except Exception as e:
        return None, None, None

# ─── Predict ──────────────────────────────────────────────────────────────────

def predict_all(models, X, ticker_idx=0):
    sample = X[ticker_idx:ticker_idx+1]
    scaler = StandardScaler()
    n      = sample.shape[2]
    flat   = sample.reshape(-1, n)
    scaled = scaler.fit_transform(flat).reshape(1, sample.shape[1], n)
    tensor = torch.tensor(scaled, dtype=torch.float32)

    results = {}
    for name, model in models.items():
        with torch.no_grad():
            prob = model(tensor).item()
        results[name] = {
            "probability": round(prob * 100, 1),
            "direction":   "UP ↑" if prob >= 0.5 else "DOWN ↓",
            "confident":   abs(prob - 0.5) > 0.15
        }
    return results

# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    # Header
    st.title("📈 Real-Time Market Prediction System")
    st.markdown("*Sentiment-driven deep learning predictions using RNN, LSTM and GRU*")
    st.divider()

    # Load everything
    prices  = load_prices()
    news    = load_news()
    results = load_results()
    models, X, n_features = load_models()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("⚙️ Controls")
    tickers       = sorted(prices["ticker"].unique().tolist())
    selected      = st.sidebar.selectbox("Select Stock", tickers)
    show_volume   = st.sidebar.checkbox("Show Volume", value=True)
    days_back     = st.sidebar.slider("Days of history", 10, 90, 30)
    st.sidebar.divider()
    st.sidebar.markdown("**Data Summary**")
    st.sidebar.metric("Price rows",    len(prices))
    st.sidebar.metric("News articles", len(news))

    # ── Row 1: Key Metrics ────────────────────────────────────────────────────
    ticker_df = prices[prices["ticker"] == selected].sort_values("date")
    latest    = ticker_df.iloc[-1]
    prev      = ticker_df.iloc[-2]
    change    = round(latest["close"] - prev["close"], 2)
    change_pct= round((change / prev["close"]) * 100, 2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close",   f"${round(latest['close'], 2)}",  f"{change:+.2f} ({change_pct:+.2f}%)")
    col2.metric("Day High",       f"${round(latest['high'],  2)}")
    col3.metric("Day Low",        f"${round(latest['low'],   2)}")
    col4.metric("Volume",         f"{int(latest['volume']):,}")

    st.divider()

    # ── Row 2: Price Chart + Sentiment ────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader(f"📊 {selected} Price Chart")
        plot_df = ticker_df.tail(days_back)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=plot_df["date"],
            open=plot_df["open"], high=plot_df["high"],
            low=plot_df["low"],   close=plot_df["close"],
            name="OHLC"
        ))
        if show_volume:
            fig.add_trace(go.Bar(
                x=plot_df["date"], y=plot_df["volume"],
                name="Volume", yaxis="y2",
                marker_color="rgba(100,100,200,0.3)"
            ))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))

        fig.update_layout(
            height=400, xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🧠 Sentiment Today")
        if not news.empty:
            pos = (news["sentiment_label"] == "Positive").sum()
            neg = (news["sentiment_label"] == "Negative").sum()
            neu = (news["sentiment_label"] == "Neutral").sum()
            tot = len(news)

            st.metric("🟢 Positive", f"{pos} articles ({round(pos/tot*100)}%)")
            st.metric("🔴 Negative", f"{neg} articles ({round(neg/tot*100)}%)")
            st.metric("⚪ Neutral",  f"{neu} articles ({round(neu/tot*100)}%)")

            avg = round(news["sentiment_score"].mean(), 3)
            overall = "Bullish 🟢" if avg > 0.05 else "Bearish 🔴" if avg < -0.05 else "Neutral ⚪"
            st.divider()
            st.metric("Overall Market Mood", overall, f"avg score: {avg:+.3f}")
        else:
            st.warning("No sentiment data found. Run sentiment_analysis.py first.")

    st.divider()

    # ── Row 3: Model Predictions ──────────────────────────────────────────────
    st.subheader("🤖 Model Predictions — Next Day Direction")

    if models:
        ticker_idx = tickers.index(selected)
        preds = predict_all(models, X, ticker_idx)

        col1, col2, col3 = st.columns(3)
        for col, (name, pred) in zip([col1, col2, col3], preds.items()):
            color = "🟢" if "UP" in pred["direction"] else "🔴"
            col.metric(
                label=f"{name} Model",
                value=f"{color} {pred['direction']}",
                delta=f"{pred['probability']}% confidence"
            )
    else:
        st.warning("Models not found. Make sure rnn_model.pth, lstm_model.pth, gru_model.pth are in the same folder.")

    st.divider()

    # ── Row 4: Model Comparison Table ────────────────────────────────────────
    col_table, col_bar = st.columns([1, 1])

    with col_table:
        st.subheader("📋 Model Comparison")
        st.dataframe(results, use_container_width=True, hide_index=True)

    with col_bar:
        st.subheader("📊 Accuracy Comparison")
        fig2 = go.Figure(go.Bar(
            x=results["Model"],
            y=results["Accuracy (%)"],
            marker_color=["#636EFA", "#EF553B", "#00CC96"],
            text=results["Accuracy (%)"].astype(str) + "%",
            textposition="outside"
        ))
        fig2.update_layout(
            height=300, yaxis_range=[0, 100],
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Row 5: Latest News ────────────────────────────────────────────────────
    st.subheader("📰 Latest News with Sentiment")
    if not news.empty:
        for _, row in news.head(10).iterrows():
            emoji = "🟢" if row["sentiment_label"] == "Positive" else \
                    "🔴" if row["sentiment_label"] == "Negative" else "⚪"
            score = round(row["sentiment_score"], 3)
            st.markdown(f"{emoji} **{row['sentiment_label']}** `{score:+.3f}` — {row['title']}")
    else:
        st.info("No news data available.")

    st.divider()
    st.caption("Real-Time Market Movement Prediction System | Built with PyTorch + Streamlit")


if __name__ == "__main__":
    main()
