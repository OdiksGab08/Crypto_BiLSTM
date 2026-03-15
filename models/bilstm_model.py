# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import os

# Import configuration for paths and training parameters
from config.config import MODEL_SAVE_PATH, EPOCHS, BATCH_SIZE, RANDOM_SEED

# Set random seed for reproducibility
import tensorflow as tf
import numpy as np
import random

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================================
# FUNCTION: build_bilstm_model
# ============================================================

def build_bilstm_model(input_shape):
    """
    Builds and compiles a Bidirectional LSTM model for time-series prediction.

    Parameters
    ----------
    input_shape : tuple
        Shape of input sequences: (timesteps, features)

    Returns
    -------
    model : tf.keras.Model
        Compiled Bidirectional LSTM model
    """

    # Initialize a Sequential model
    model = Sequential()

    # First Bidirectional LSTM layer
    # - 64 units
    # - return_sequences=True to feed the next LSTM layer
    # - input_shape = (timesteps, features)
    model.add(
        Bidirectional(
            LSTM(64, return_sequences=True),
            input_shape=input_shape
        )
    )

    # Add dropout for regularization (reduce overfitting)
    model.add(Dropout(0.2))

    # Second Bidirectional LSTM layer
    # - 32 units
    # - return_sequences=False because this is the last LSTM layer
    model.add(Bidirectional(LSTM(32)))

    # Output layer
    # Dense layer with 1 unit predicts the next price
    model.add(Dense(1))

    # Compile the model
    # - optimizer: 'adam' is widely used for regression tasks
    # - loss: mean squared error (MSE) for price prediction
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Print model summary to check architecture
    model.summary()

    return model

# ============================================================
# FUNCTION: train_bilstm_model
# ============================================================

def train_bilstm_model(model, X_train, y_train, X_val=None, y_val=None):
    """
    Trains the Bidirectional LSTM model on preprocessed data.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled Bi-LSTM model
    X_train : np.ndarray
        Training sequences (samples, timesteps, features)
    y_train : np.ndarray
        Training targets (next prices)
    X_val : np.ndarray, optional
        Validation sequences
    y_val : np.ndarray, optional
        Validation targets

    Returns
    -------
    history : tf.keras.callbacks.History
        Training history object
    """

    # Early stopping callback
    # Stops training if validation loss doesn't improve for 10 epochs
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Fit the model
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val) if X_val is not None else None,
        callbacks=[early_stop],
        shuffle=False
    )

    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Save the trained model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    return history

# ============================================================
# TEST BLOCK (optional)
# ============================================================

if __name__ == "__main__":

    # This test block is for developers to quickly test the model build
    # Example input shape: 60 timesteps, 1 feature
    sample_input_shape = (60, 1)

    # Build model
    model = build_bilstm_model(sample_input_shape)

    # Note: Actual training requires X_train, y_train arrays
    print("Bi-LSTM model built successfully!")