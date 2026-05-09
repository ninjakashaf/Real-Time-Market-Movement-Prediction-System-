"""
Train RNN, LSTM and GRU Models
================================
Project: Real-Time Market Movement Prediction System
Phase 4: Train and compare 3 deep learning models

Requirements:
    pip install torch scikit-learn numpy pandas matplotlib

Run AFTER build_dataset.py
"""
import mlflow
import mlflow.pytorch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── Config ───────────────────────────────────────────────────────────────────

EPOCHS     = 30
BATCH_SIZE = 16
LR         = 0.001
HIDDEN     = 64
LAYERS     = 2
DROPOUT    = 0.3
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Step 1: Load sequences ───────────────────────────────────────────────────

def load_sequences():
    X = np.load("X_sequences.npy", allow_pickle=True).astype(np.float32)
    y = np.load("y_labels.npy",    allow_pickle=True).astype(np.float32)
    print(f"Loaded X: {X.shape}, y: {y.shape}")
    return X, y


# ─── Step 2: Normalise + Split ────────────────────────────────────────────────

def prepare_data(X, y):
    n_samples, n_steps, n_features = X.shape

    # Normalise features
    X_flat   = X.reshape(-1, n_features)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat).reshape(n_samples, n_steps, n_features)

    # Train / test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=SEED, shuffle=True
    )

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE, shuffle=True
    )

    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    return train_loader, X_test_t, y_test_t, n_features


# ─── Step 3: Model Definitions ───────────────────────────────────────────────

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.rnn = nn.RNN(
            input_size, hidden, layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze()


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden, layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden, layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze()


# ─── Step 4: Train one model ──────────────────────────────────────────────────

def train_model(model, train_loader, name):
    print(f"\nTraining {name}...")
    print("-" * 40)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    losses    = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:>2}/{EPOCHS}  Loss: {avg_loss:.4f}")

    return losses


# ─── Step 5: Evaluate one model ───────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, name):
    model.eval()
    with torch.no_grad():
        preds_raw = model(X_test).numpy()

    preds  = (preds_raw >= 0.5).astype(int)
    y_true = y_test.numpy().astype(int)

    acc = round(accuracy_score(y_true, preds) * 100, 1)
    f1  = round(f1_score(y_true, preds, zero_division=0), 3)

    print(f"\n{name} Results:")
    print(f"  Accuracy : {acc}%")
    print(f"  F1-Score : {f1}")
    print(classification_report(y_true, preds,
          target_names=["DOWN", "UP"], zero_division=0))

    return {"Model": name, "Accuracy (%)": acc, "F1-Score": f1}


# ─── Step 6: Plot training curves ─────────────────────────────────────────────

def plot_losses(all_losses):
    plt.figure(figsize=(10, 5))
    for name, losses in all_losses.items():
        plt.plot(losses, label=name)
    plt.title("Training Loss — RNN vs LSTM vs GRU")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss.png")
    print("\nTraining loss chart saved: training_loss.png")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_training():
    print("=" * 60)
    print("Phase 4: Training RNN, LSTM, GRU Models with MLflow")
    print("=" * 60)

    X, y = load_sequences()
    train_loader, X_test, y_test, n_features = prepare_data(X, y)

    # Define all 3 models
    models = {
        "RNN":  RNNModel(n_features),
        "LSTM": LSTMModel(n_features),
        "GRU":  GRUModel(n_features),
    }

    all_losses = {}
    all_results = []

    # 1. Set the MLflow Experiment Name
    mlflow.set_experiment("Market_Movement_Prediction")

    # Train and evaluate each model using MLflow tracking
    for name, model in models.items():
        # 2. Start an MLflow run for each model
        with mlflow.start_run(run_name=name):
            
            # 3. Log the hyperparameters
            mlflow.log_params({
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LR,
                "hidden_dim": HIDDEN,
                "layers": LAYERS,
                "dropout": DROPOUT,
                "seed": SEED
            })

            losses = train_model(model, train_loader, name)
            all_losses[name] = losses

            result = evaluate_model(model, X_test, y_test, name)
            all_results.append(result)

            # 4. Log the evaluation metrics
            mlflow.log_metric("accuracy", result["Accuracy (%)"])
            mlflow.log_metric("f1_score", result["F1-Score"])
            mlflow.log_metric("final_train_loss", losses[-1])

            # 5. Save the trained model locally AND to MLflow
            torch.save(model.state_dict(), f"{name.lower()}_model.pth")
            mlflow.pytorch.log_model(model, artifact_path=f"{name.lower()}_model")
            print(f"  Model saved locally and logged to MLflow!")

    # Plot training curves
    plot_losses(all_losses)

    # Final comparison table
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("Accuracy (%)", ascending=False)

    print("\n" + "=" * 60)
    print("FINAL COMPARISON TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]["Model"]
    print(f"\nBest model: {best}")
    print("=" * 60)

    # Save results to CSV
    results_df.to_csv("model_results.csv", index=False)
    print("Results saved: model_results.csv")

    return results_df


if __name__ == "__main__":
    run_training()
