# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Import configuration variables
from config.config import LOOKBACK_DAYS, RANDOM_SEED


# ============================================================
# FUNCTION: prepare_data
# ============================================================

def prepare_data(df, feature_col="price", lookback=LOOKBACK_DAYS, split_ratio=0.8):
    """
    Prepares the cryptocurrency data for training a Bidirectional LSTM.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing historical price data with a column 'price'.

    feature_col : str
        The column name containing the price data.

    lookback : int
        Number of previous time steps to include in each input sequence.

    split_ratio : float
        Fraction of data to use for training (0 < split_ratio < 1).

    Returns
    -------
    X_train, y_train : np.ndarray
        Training sequences and targets.

    X_test, y_test : np.ndarray
        Testing sequences and targets.

    scaler : sklearn.preprocessing.MinMaxScaler
        The fitted scaler object (needed to inverse-transform predictions).
    """

    # --------------------------------------------------------
    # Step 1: Extract the price data
    # --------------------------------------------------------
    # Convert the selected column into a numpy array of floats
    data = df[[feature_col]].values.astype(float)

    # --------------------------------------------------------
    # Step 2: Normalize the data
    # --------------------------------------------------------
    # MinMaxScaler scales values to range [0, 1]
    # This ensures that large numbers do not dominate the neural network
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # --------------------------------------------------------
    # Step 3: Create sequences for LSTM input
    # --------------------------------------------------------
    # For each sequence, take 'lookback' previous prices as X
    # and the next price as y
    X = []
    y = []

    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])  # sequence of previous prices
        y.append(scaled_data[i, 0])               # target: next price

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # --------------------------------------------------------
    # Step 4: Reshape X for LSTM input
    # --------------------------------------------------------
    # LSTM expects input shape: (samples, timesteps, features)
    # Currently X has shape (samples, timesteps)
    # Reshape to (samples, timesteps, features=1)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # --------------------------------------------------------
    # Step 5: Split into training and testing sets
    # --------------------------------------------------------
    # Training data: first 'split_ratio' fraction
    split_index = int(len(X) * split_ratio)

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]

    # --------------------------------------------------------
    # Step 6: Return prepared data
    # --------------------------------------------------------
    return X_train, y_train, X_test, y_test, scaler


# ============================================================
# FUNCTION: create_future_sequences
# ============================================================

def create_future_sequences(data, lookback=LOOKBACK_DAYS):
    """
    Prepares the latest sequence of prices for predicting future prices.
    This is used for making predictions with the trained model.

    Parameters
    ----------
    data : pandas.DataFrame or np.ndarray
        Historical price data (latest prices).

    lookback : int
        Number of previous steps to include in the input sequence.

    Returns
    -------
    np.ndarray
        3D array of shape (1, lookback, 1) ready for model.predict()
    """

    # Convert to numpy array if it is a DataFrame
    if isinstance(data, pd.DataFrame):
        prices = data["price"].values
    else:
        prices = data

    # Take the last 'lookback' prices
    last_sequence = prices[-lookback:]

    # Reshape to (1, timesteps, features) for model prediction
    last_sequence = np.reshape(last_sequence, (1, lookback, 1))

    return last_sequence