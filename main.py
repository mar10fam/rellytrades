import yfinance as yf
import pandas as pd
import os

# The stock ticker symbol to download data for
TICKER = "TSLA"

# Directory where CSV files will be saved
DATA_DIR = "data"

# yfinance allows for 60 days of free 5-minute data
PERIOD = "60d"

# Need two timeframes
# - 5m candles: identify high and low of the day
# - 1m candles: identify FVG breakouts and confirmation signals
INTERVALS = ["5m", "1m"]

def download_stock_data(ticker: str, period: str, interval: str = "5m") -> pd.DataFrame:
    """
    Downloads historical stock data from Yahoo Finance for a single interval.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")
        period: How far back to download (e.g., "60d" for 60 days)
        interval: Candle size (e.g., "5m" for 5-minute candles, "1m" for 1-minute candles)

    Returns:
        A pandas DataFrame with columns: Open, High, Low, Close, Volume
    """
    print(f"Downloading {interval} data for {ticker} over the past {period}...")

    # Download data using yfinance
    data = yf.download(ticker, period=period, interval=interval)

    # Check if data was downloaded successfully
    if data.empty:
        print(f"Warning: No data downloaded for {ticker} with interval {interval}.")
        return pd.DataFrame()  # Return an empty DataFrame
    
    print(f"Downloaded {len(data)} rows of {interval} data for {ticker}.")
    print(f"Data range: {data.index[0]} to {data.index[-1]}")

    return data

def download_all_intervals(ticker: str, period: str = PERIOD) -> dict[str, pd.DataFrame]:
    """
    Downloads stock data for all required timeframes (5m and 1m).

    We need both because:
    - 5m candles are used to mark the opening range (high/low of first candle at 9:30 AM)
    - 1m candles are used to detect FVG breakouts and confirmation signals with more granularity

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        period: How far back to download (e.g., "60d")

    Returns:
        A dictionary mapping interval strings to their DataFrames.
        Example: {"5m": DataFrame, "1m": DataFrame}
    """
    all_data = {}

    for interval in INTERVALS:
        print(f"\n--- {interval} candles ---")
        data = download_stock_data(ticker, period=period, interval=interval)

        if not data.empty:
            all_data[interval] = data

    return all_data
    
def save_to_csv(data: pd.DataFrame, ticker: str, interval: str) -> str:
    """
    Saves the downloaded stock data to a CSV file in the data/ directory.

    Args:
        data: The DataFrame of stock data to save
        ticker: Stock ticker symbol (used in the filename)
        interval: The candle interval (e.g., "5m", "1m") — used in the filename

    Returns:
        The file path where the CSV was saved
    """

def load_from_csv(ticker: str, interval: str) -> pd.DataFrame:
    """
    Loads previously saved stock data from a CSV file.

    Args:
        ticker: Stock ticker symbol (used to find the filename)
        interval: The candle interval (e.g., "5m", "1m")

    Returns:
        A pandas DataFrame with the stock data, indexed by datetime
    """

def inspect_data(data: pd.DataFrame, label: str = "") -> None:
    """
    Prints a summary of the stock data for quick inspection.

    Args:
        data: The DataFrame of stock data to inspect
        label: Optional label to identify which dataset this is (e.g., "5m", "1m")
    """

# === Main entry point ===
if __name__ == "__main__":
    all_data = download_all_intervals(TICKER)

    print(all_data)