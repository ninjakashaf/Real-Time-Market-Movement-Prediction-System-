import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import torch
import numpy as np
from train_models import RNNModel  # Importing your teammate's model architecture

app = FastAPI(title="Market Movement Predictor API")

# --- 1. Load the Model & Data ---
try:
    # Load a sample of the data to get the number of features
    X_sample = np.load("X_sequences.npy", allow_pickle=True).astype(np.float32)
    n_features = X_sample.shape[2]
    
    # Initialize the model and load the trained weights you just saved
    model = RNNModel(input_size=n_features)
    model.load_state_dict(torch.load("rnn_model.pth", weights_only=True))
    model.eval()
    print("RNN Model loaded successfully!")
except Exception as e:
    print(f"Error loading model or data: {e}")

# --- 2. Build the API Endpoint ---
@app.get("/predict")
def predict_market():
    # For testing, we grab a random sequence from the dataset
    random_idx = np.random.randint(0, len(X_sample))
    sample_sequence = X_sample[random_idx]
    
    # Convert to PyTorch tensor and add batch dimension
    tensor_sequence = torch.tensor(sample_sequence, dtype=torch.float32).unsqueeze(0)
    
    # Make the prediction
    with torch.no_grad():
        prediction_raw = model(tensor_sequence).item()
        
    # Threshold at 0.5 (1 = UP, 0 = DOWN)
    market_direction = "UP 📈" if prediction_raw >= 0.5 else "DOWN 📉"
    
    return {
        "sequence_index": random_idx,
        "raw_probability": round(prediction_raw, 4),
        "prediction": market_direction
    }

# --- 3. Build the Simple Frontend ---
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_content = """
    <html>
        <head>
            <title>Market Predictor Test</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; background-color: #f4f4f9; }
                .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); display: inline-block; }
                button { padding: 10px 20px; font-size: 16px; cursor: pointer; background-color: #007bff; color: white; border: none; border-radius: 5px; }
                button:hover { background-color: #0056b3; }
                #result { margin-top: 20px; font-size: 20px; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Real-Time Market Movement Predictor</h2>
                <p>Click the button below to pull a random time-series sequence from our test data and run it through the RNN model.</p>
                <button onclick="getPrediction()">Run Model Test</button>
                <div id="result">Waiting for test...</div>
            </div>

            <script>
                async function getPrediction() {
                    document.getElementById('result').innerText = "Running inference...";
                    const response = await fetch('/predict');
                    const data = await response.json();
                    document.getElementById('result').innerHTML = 
                        "Data Sequence ID: " + data.sequence_index + "<br>" +
                        "Raw Probability: " + data.raw_probability + "<br>" +
                        "<span style='font-size: 24px; color: " + (data.prediction.includes('UP') ? 'green' : 'red') + "'>" + 
                        "Predicted Movement: " + data.prediction + "</span>";
                }
            </script>
        </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)