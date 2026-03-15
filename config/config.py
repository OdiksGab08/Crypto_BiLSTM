# ============================================================
# COINGECKO API CONFIGURATION
# ============================================================

# Base URL for all CoinGecko API requests.
# This is the root endpoint used to access cryptocurrency market data.
# Example full endpoint used later:
# https://api.coingecko.com/api/v3/coins/bitcoin/market_chart
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"


# ============================================================
# SUPPORTED CRYPTOCURRENCIES
# ============================================================

# List of cryptocurrencies supported by the application.
# These names correspond to CoinGecko's coin IDs.
#
# Example:
# https://api.coingecko.com/api/v3/coins/bitcoin
#
# Adding more coins here will automatically allow the system
# to fetch their data and run predictions.
SUPPORTED_COINS = [
    "bitcoin",
    "ethereum",
    "solana",
    "tether",
    "binancecoin",
    "cardano"
]


# ============================================================
# DATA COLLECTION SETTINGS
# ============================================================

# Number of days of historical data to retrieve from the API.
# This will be used to train the model and generate predictions.
#
# Example:
# If HISTORICAL_DAYS = 365
# The system will fetch 1 year of crypto price data.
HISTORICAL_DAYS = 365


# Currency used for price comparison.
# CoinGecko allows different currencies (usd, eur, gbp, etc.)
# For this project we use USD since it is the most common
# reference currency in crypto markets.
VS_CURRENCY = "usd"


# ============================================================
# MACHINE LEARNING PARAMETERS
# ============================================================

# LOOKBACK_DAYS defines how many previous time steps the model
# uses to predict the next value.
#
# Example:
# If LOOKBACK_DAYS = 60
# The model uses the previous 60 days of prices
# to predict the next day's price.
LOOKBACK_DAYS = 60


# Number of days the model should predict into the future.
#
# Example:
# If PREDICTION_DAYS = 7
# The model will generate predictions for the next 7 days.
PREDICTION_DAYS = 7


# ============================================================
# MODEL TRAINING PARAMETERS
# ============================================================

# Number of training iterations over the dataset.
# More epochs allow the model to learn more patterns,
# but too many can cause overfitting.
EPOCHS = 20


# Batch size determines how many samples are processed
# before updating the neural network weights.
#
# Larger batch sizes speed up training but require
# more memory.
BATCH_SIZE = 32


# ============================================================
# FILE PATH CONFIGURATION
# ============================================================

# Path where trained models will be saved.
# This allows the Flask application to load the trained model
# later for predictions without retraining it.
MODEL_SAVE_PATH = "models/trained_models/bilstm_crypto_model.h5"


# Path where fetched cryptocurrency data can be stored.
# Storing data locally helps avoid repeatedly calling
# the API during development and testing.
DATA_SAVE_PATH = "data/crypto_data.csv"


# ============================================================
# RANDOM SEED (FOR REPRODUCIBILITY)
# ============================================================

# Machine learning models often involve random processes
# such as weight initialization and data shuffling.
#
# Setting a random seed ensures that results are
# reproducible when training the model multiple times.
RANDOM_SEED = 42


# ============================================================
# FLASK APPLICATION SETTINGS
# ============================================================

# Host address where the Flask application will run.
# "127.0.0.1" means the server runs locally on your machine.
FLASK_HOST = "127.0.0.1"


# Port number for the Flask server.
# The default Flask port is 5000.
FLASK_PORT = 5000


# Debug mode allows automatic server reloads when code changes.
# It is useful during development but should be disabled
# in production deployments.
FLASK_DEBUG = True