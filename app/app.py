# ============================================================
# IMPORT LIBRARIES
# ============================================================

from flask import Flask, render_template, request, jsonify  # Flask core
from utils.data_fetcher import fetch_crypto_data           # Fetch crypto data
from utils.preprocessing import prepare_data, create_future_sequences  # Preprocess
from tensorflow.keras.models import load_model            # Load trained Bi-LSTM
import numpy as np
import pandas as pd
from config.config import (
    SUPPORTED_COINS,
    LOOKBACK_DAYS,
    PREDICTION_DAYS,
    MODEL_SAVE_PATH
)

# ============================================================
# INITIALIZE FLASK APP
# ============================================================

app = Flask(__name__)  # Initialize Flask

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

# Load the trained Bi-LSTM model from disk
# This is done once at server start to avoid reloading every request
try:
    model = load_model(MODEL_SAVE_PATH)
    print(f"Successfully loaded model from {MODEL_SAVE_PATH}")
except Exception as e:
    print("Error loading Bi-LSTM model. Make sure it exists!")
    print(e)
    model = None

# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/")
def home():
    """
    Home page route.
    Renders the dashboard HTML where users can select coins
    and view predictions.
    """
    return render_template("index.html")  # Flask looks in templates/ by default

# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts POST requests from the frontend to make crypto predictions.

    Request JSON format:
    {
        "crypto": "bitcoin"
    }

    Returns JSON:
    {
        "crypto": "bitcoin",
        "current_price": 45000,
        "next_prediction": 45200,
        "forecast": [...],
        "dates": [...]
    }
    """

    # Step 1: Parse JSON from frontend
    data = request.get_json()
    crypto = data.get("crypto")

    # Step 2: Validate cryptocurrency
    if crypto not in SUPPORTED_COINS:
        return jsonify({"error": "Unsupported cryptocurrency"}), 400

    # Step 3: Fetch historical price data
    df = fetch_crypto_data(coin_id=crypto)
    if df is None or df.empty:
        return jsonify({"error": "Failed to fetch crypto data"}), 500

    # Step 4: Preprocess data into sequences for Bi-LSTM
    # We use create_future_sequences to get the latest lookback days
    latest_sequence = create_future_sequences(df, lookback=LOOKBACK_DAYS)

    # Step 5: Make predictions for the next day and future days
    forecast_scaled = []

    # Copy sequence to modify iteratively
    input_seq = latest_sequence.copy()

    # Loop to predict next PREDICTION_DAYS sequentially
    for _ in range(PREDICTION_DAYS):
        pred = model.predict(input_seq, verbose=0)  # Predict next price
        forecast_scaled.append(pred[0, 0])          # Store the prediction

        # Append prediction to sequence and remove first value (sliding window)
        input_seq = np.append(input_seq[:, 1:, :], [[ [pred[0, 0]] ]], axis=1)

    # Step 6: Inverse scale predictions to original price range
    # We fit MinMaxScaler on the full price data for simplicity
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(df[["price"]].values.reshape(-1, 1))
    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

    # Step 7: Current price (latest in dataset)
    current_price = df["price"].values[-1]

    # Step 8: Prepare dates for forecast
    last_date = df["date"].values[-1]
    dates = pd.date_range(start=last_date, periods=PREDICTION_DAYS + 1, freq="D")[1:]
    dates = [str(d.date()) for d in dates]

    # Step 9: Return JSON response
    return jsonify({
        "crypto": crypto,
        "current_price": float(current_price),
        "next_prediction": float(forecast[0]),
        "forecast": forecast.tolist(),
        "dates": dates
    })

# ============================================================
# RUN FLASK APP
# ============================================================

if __name__ == "__main__":
    # Debug=True allows live reload on code changes
    app.run(host="127.0.0.1", port=5000, debug=True)