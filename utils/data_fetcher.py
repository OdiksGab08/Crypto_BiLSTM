# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

# requests
# Used to send HTTP requests to the CoinGecko API
import requests

# pandas
# Used for data manipulation and tabular data structures
import pandas as pd

# datetime
# Helps convert timestamps into human-readable dates
from datetime import datetime

# Import configuration variables from our config file
# This allows us to reuse API settings without hardcoding them
from config.config import (
    COINGECKO_BASE_URL,
    HISTORICAL_DAYS,
    VS_CURRENCY,
    DATA_SAVE_PATH
)


# ============================================================
# FUNCTION: fetch_crypto_data
# ============================================================

def fetch_crypto_data(coin_id="bitcoin", days=HISTORICAL_DAYS):
    """
    Fetch historical cryptocurrency price data from CoinGecko.

    Parameters
    ----------
    coin_id : str
        The cryptocurrency identifier used by CoinGecko.
        Examples:
        - "bitcoin"
        - "ethereum"
        - "solana"

    days : int
        Number of days of historical data to retrieve.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing:
        - date
        - price
    """

    # --------------------------------------------------------
    # Construct the API endpoint URL
    # --------------------------------------------------------

    # CoinGecko endpoint for historical market data
    # Example:
    # https://api.coingecko.com/api/v3/coins/bitcoin/market_chart
    endpoint = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"

    # --------------------------------------------------------
    # Define API request parameters
    # --------------------------------------------------------

    params = {
        "vs_currency": VS_CURRENCY,  # Currency used for pricing (USD)
        "days": days                 # Number of days of historical data
    }

    # --------------------------------------------------------
    # Send request to the API
    # --------------------------------------------------------

    try:
        response = requests.get(endpoint, params=params)

        # Check if the API request was successful
        response.raise_for_status()

    except requests.exceptions.RequestException as error:

        # If an error occurs (network issue, invalid coin, etc.)
        print("Error fetching data from CoinGecko API")
        print(error)

        return None

    # --------------------------------------------------------
    # Convert API response into JSON format
    # --------------------------------------------------------

    data = response.json()

    # --------------------------------------------------------
    # Extract price data
    # --------------------------------------------------------

    """
    CoinGecko returns price data in this format:

    "prices": [
        [timestamp, price],
        [timestamp, price],
        ...
    ]

    Example:

    [
        [1710000000000, 45000.23],
        [1710086400000, 45210.10]
    ]
    """

    prices = data["prices"]

    # --------------------------------------------------------
    # Convert raw data into a pandas DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(prices, columns=["timestamp", "price"])

    # --------------------------------------------------------
    # Convert timestamps into readable dates
    # --------------------------------------------------------

    # CoinGecko timestamps are in milliseconds
    # We convert them into Python datetime objects
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")

    # --------------------------------------------------------
    # Remove the original timestamp column
    # --------------------------------------------------------

    df = df.drop(columns=["timestamp"])

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    df = df[["date", "price"]]

    # --------------------------------------------------------
    # Reset DataFrame index
    # --------------------------------------------------------

    df.reset_index(drop=True, inplace=True)

    # --------------------------------------------------------
    # Print success message
    # --------------------------------------------------------

    print(f"Successfully fetched {len(df)} records for {coin_id}")

    return df


# ============================================================
# FUNCTION: save_data_to_csv
# ============================================================

def save_data_to_csv(df, file_path=DATA_SAVE_PATH):
    """
    Save a pandas DataFrame to a CSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to save

    file_path : str
        Path where the CSV file will be stored
    """

    try:
        df.to_csv(file_path, index=False)

        print(f"Data successfully saved to {file_path}")

    except Exception as error:

        print("Error saving data to CSV")
        print(error)


# ============================================================
# MAIN TEST BLOCK
# ============================================================

"""
This section runs only when this file is executed directly.

It allows developers to test the data fetching functionality
without running the entire application.
"""

if __name__ == "__main__":

    # Fetch Bitcoin historical data
    crypto_df = fetch_crypto_data("bitcoin")

    # If data retrieval was successful
    if crypto_df is not None:

        # Display first few rows
        print(crypto_df.head())

        # Save dataset locally
        save_data_to_csv(crypto_df)