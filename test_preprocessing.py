# test_preprocessing.py

import pandas as pd
from utils.preprocessing import prepare_data, create_future_sequences

# Step 1: Load sample data
# Use your locally saved CSV from data_fetcher.py
df = pd.read_csv("data/crypto_data.csv")

print("First 5 rows of raw data:")
print(df.head())

# Step 2: Prepare data for Bi-LSTM
X_train, y_train, X_test, y_test, scaler = prepare_data(df)

# Step 3: Print shapes to verify sequence creation
print("\nShapes after preprocessing:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Step 4: Check first training sequence and target
print("\nFirst training sequence (scaled):")
print(X_train[0])
print("Corresponding target (scaled):", y_train[0])

# Step 5: Prepare latest sequence for future prediction
latest_seq = create_future_sequences(df)

print("\nLatest sequence for prediction (shape):", latest_seq.shape)
print("Latest sequence values (scaled or raw depending on preprocessing):")
print(latest_seq)