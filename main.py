import os
import time
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from polygon import RESTClient


# Load environment variables from .env file
# This reads the .env file and makes its values available via os.getenv()
load_dotenv()


# === Configuration ===

# The stock ticker symbol to download data for
TICKER = "AAPL"

# Directory where CSV files will be saved
DATA_DIR = "data"

# How far back to download data (in days)
# Polygon free tier provides up to 2 years of historical intraday data
LOOKBACK_DAYS = 730  # ~2 years

# Polygon free tier rate limit: 5 API calls per minute.
# The Polygon client auto-paginates large requests behind the scenes,
# and each paginated page counts as 1 API call. To stay under the limit,
# we download in small date-range chunks and pause between each one.
#
# 1-minute data has 5x more raw candles than 5-minute data, which means
# more internal pagination per chunk. So we use smaller chunks for 1m.
RATE_LIMIT_PAUSE_SECONDS = 15  # Pause between chunks (safely under 5 req/min)
MAX_RETRIES = 3  # How many times to retry a chunk after a 429 error
RETRY_WAIT_SECONDS = 65  # How long to wait after a 429 (resets the 1-min rate window)

# Chunk sizes per interval — how many days of data to request per API call.
# 1m data needs smaller chunks because each day has 5x more candles,
# which causes more internal pagination and burns through API calls faster.
CHUNK_SIZE_DAYS = {
    "5m": 30,  # 30 days of 5m data fits comfortably in 1 API page
    "1m": 7,   # 7 days of 1m data keeps pagination minimal
}

# We need two timeframes:
# - 5m candles: to identify the opening range (first candle high/low after 9:30 AM ET)
# - 1m candles: to detect FVG breakouts and confirmation signals with more precision
INTERVAL_CONFIG = {
    "5m": {"multiplier": 5, "timespan": "minute"},
    "1m": {"multiplier": 1, "timespan": "minute"},
}


def get_polygon_client() -> RESTClient:
    """
    Creates and returns a Polygon API client using the API key from .env.

    Returns:
        An authenticated RESTClient instance

    Raises:
        ValueError: If the POLYGON_API_KEY is not found in .env
    """
    api_key = os.getenv("POLYGON_API_KEY")

    if not api_key:
        raise ValueError(
            "POLYGON_API_KEY not found. "
            "Make sure you have a .env file with POLYGON_API_KEY=your_key_here"
        )

    return RESTClient(api_key)


def download_stock_data_chunk(
    client: RESTClient,
    ticker: str,
    multiplier: int,
    timespan: str,
    from_date: str,
    to_date: str,
) -> list[dict]:
    """
    Downloads a single chunk of historical stock data from Polygon.io.

    This is a helper called by download_stock_data() for each chunk.
    Returns raw bar data as a list of dicts so chunks can be combined
    efficiently before creating the final DataFrame.

    Downloads all candles (full trading day) — filtering to specific
    time windows (e.g., first 2 hours) happens later in the strategy layer.

    Args:
        client: An authenticated Polygon RESTClient
        ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")
        multiplier: The size of the candle (e.g., 5 for 5-minute candles)
        timespan: The unit of time ("minute", "hour", "day", etc.)
        from_date: Start date in "YYYY-MM-DD" format
        to_date: End date in "YYYY-MM-DD" format

    Returns:
        A list of dictionaries, each with keys: Datetime, Open, High, Low, Close, Volume
    """
    # Polygon's list_aggs returns an iterator of aggregate bars.
    aggs = client.list_aggs(
        ticker=ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_=from_date,
        to=to_date,
        limit=50000,
    )

    # Convert the response into a list of dicts
    bars = []
    for agg in aggs:
        bars.append({
            "Datetime": pd.Timestamp(agg.timestamp, unit="ms", tz="US/Eastern"),
            "Open": agg.open,
            "High": agg.high,
            "Low": agg.low,
            "Close": agg.close,
            "Volume": agg.volume,
        })

    return bars


def download_stock_data_chunk_with_retry(
    client: RESTClient,
    ticker: str,
    multiplier: int,
    timespan: str,
    from_date: str,
    to_date: str,
) -> list[dict]:
    """
    Wrapper around download_stock_data_chunk that retries on rate limit errors.

    If Polygon returns a 429 (too many requests), we wait for the rate limit
    window to reset (~1 minute) and try again. This prevents the entire
    download from crashing just because one chunk hit the limit.

    Args:
        Same as download_stock_data_chunk

    Returns:
        A list of dictionaries, each with keys: Datetime, Open, High, Low, Close, Volume

    Raises:
        Exception: If all retries are exhausted
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return download_stock_data_chunk(
                client=client,
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_date=from_date,
                to_date=to_date,
            )
        except Exception as e:
            # Check if this is a rate limit error (429)
            if "429" in str(e):
                if attempt < MAX_RETRIES:
                    print(f"\n  Rate limited! Waiting {RETRY_WAIT_SECONDS}s for limit to reset "
                          f"(retry {attempt}/{MAX_RETRIES})...", flush=True)
                    time.sleep(RETRY_WAIT_SECONDS)
                else:
                    print(f"\n  Rate limited {MAX_RETRIES} times in a row. Giving up on this chunk.")
                    raise
            else:
                # Not a rate limit error — don't retry, just raise it
                raise


def download_stock_data(
    client: RESTClient,
    ticker: str,
    multiplier: int,
    timespan: str,
    from_date: str,
    to_date: str,
    interval_label: str,
) -> pd.DataFrame:
    """
    Downloads historical stock data from Polygon.io in chunks to respect
    the free tier rate limit (5 API calls/minute).

    Chunk size varies by interval:
    - 5m data: 30-day chunks (fewer raw candles per day = less pagination)
    - 1m data: 7-day chunks (more raw candles per day = needs smaller chunks)

    Downloads full trading days. Filtering to specific time windows
    (e.g., first 2 hours after open) happens in the strategy layer.

    If a chunk hits a 429 rate limit, it waits and retries automatically.

    Args:
        client: An authenticated Polygon RESTClient
        ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")
        multiplier: The size of the candle (e.g., 5 for 5-minute candles)
        timespan: The unit of time ("minute", "hour", "day", etc.)
        from_date: Start date in "YYYY-MM-DD" format
        to_date: End date in "YYYY-MM-DD" format
        interval_label: Human-readable label like "5m" or "1m" (for chunk size lookup)

    Returns:
        A pandas DataFrame with columns: Open, High, Low, Close, Volume
        indexed by datetime in US/Eastern timezone
    """
    print(f"Downloading {interval_label} data for {ticker} from {from_date} to {to_date}...")

    # Look up the right chunk size for this interval
    chunk_days = CHUNK_SIZE_DAYS.get(interval_label, 30)

    # Build a list of (chunk_start, chunk_end) date pairs
    chunks = []
    chunk_start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(to_date, "%Y-%m-%d")

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
        chunks.append((
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        ))
        chunk_start = chunk_end + timedelta(days=1)

    total_chunks = len(chunks)
    estimated_minutes = (total_chunks * RATE_LIMIT_PAUSE_SECONDS) / 60
    print(f"  {total_chunks} chunks ({chunk_days}-day each), ~{estimated_minutes:.0f} min estimated")

    # Download each chunk with a pause between requests
    all_bars = []
    for i, (chunk_from, chunk_to) in enumerate(chunks):
        chunk_num = i + 1
        print(f"  [{chunk_num}/{total_chunks}] {chunk_from} to {chunk_to}...", end=" ", flush=True)

        bars = download_stock_data_chunk_with_retry(
            client=client,
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_date=chunk_from,
            to_date=chunk_to,
        )

        print(f"{len(bars)} candles")
        all_bars.extend(bars)

        # Pause between chunks to stay under the rate limit
        if i < total_chunks - 1:
            print(f"  Waiting {RATE_LIMIT_PAUSE_SECONDS}s (rate limit)...", flush=True)
            time.sleep(RATE_LIMIT_PAUSE_SECONDS)

    # If no data came back, return empty
    if not all_bars:
        print(f"ERROR: No data returned for {ticker} at {interval_label}.")
        return pd.DataFrame()

    # Combine all chunks into one DataFrame
    data = pd.DataFrame(all_bars)
    data.set_index("Datetime", inplace=True)
    data.sort_index(inplace=True)

    # Remove any duplicate timestamps at chunk boundaries
    data = data[~data.index.duplicated(keep="first")]

    print(f"  Done! {len(data)} total candles for {ticker} at {interval_label}")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")

    return data


def download_all_intervals(client: RESTClient, ticker: str) -> dict[str, pd.DataFrame]:
    """
    Downloads stock data for all required timeframes (5m and 1m).

    We need both because:
    - 5m candles are used to mark the opening range (high/low of first candle at 9:30 AM)
    - 1m candles are used to detect FVG breakouts and confirmation signals with more granularity

    Downloads full trading days. Saves each interval to CSV as soon as it
    completes, so if 1m fails you don't lose the 5m data.

    Args:
        client: An authenticated Polygon RESTClient
        ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
        A dictionary mapping interval strings to their DataFrames.
        Example: {"5m": DataFrame, "1m": DataFrame}
    """
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    all_data = {}

    for interval_label, config in INTERVAL_CONFIG.items():
        print(f"\n{'=' * 60}")
        print(f"  {interval_label} candles | {from_date} to {to_date}")
        print(f"{'=' * 60}")

        data = download_stock_data(
            client=client,
            ticker=ticker,
            multiplier=config["multiplier"],
            timespan=config["timespan"],
            from_date=from_date,
            to_date=to_date,
            interval_label=interval_label,
        )

        if not data.empty:
            all_data[interval_label] = data
            # Save immediately so we don't lose progress if the next interval fails
            save_to_csv(data, ticker, interval_label)

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
    # Create the data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Build the file path (e.g., "data/AAPL_5m.csv" or "data/AAPL_1m.csv")
    file_path = os.path.join(DATA_DIR, f"{ticker}_{interval}.csv")

    # Save to CSV, including the datetime index
    data.to_csv(file_path)
    print(f"Saved data to {file_path}")

    return file_path


def load_from_csv(ticker: str, interval: str) -> pd.DataFrame:
    """
    Loads previously saved stock data from a CSV file.

    Args:
        ticker: Stock ticker symbol (used to find the filename)
        interval: The candle interval (e.g., "5m", "1m")

    Returns:
        A pandas DataFrame with the stock data, indexed by datetime
    """
    file_path = os.path.join(DATA_DIR, f"{ticker}_{interval}.csv")

    if not os.path.exists(file_path):
        print(f"ERROR: No saved data found at {file_path}")
        return pd.DataFrame()

    # Read the CSV and parse the first column (Datetime) as the index
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(data)} candles from {file_path}")

    return data


def inspect_data(data: pd.DataFrame, label: str = "") -> None:
    """
    Prints a summary of the stock data for quick inspection.

    Args:
        data: The DataFrame of stock data to inspect
        label: Optional label to identify which dataset this is (e.g., "5m", "1m")
    """
    header = f"=== Data Summary ({label}) ===" if label else "=== Data Summary ==="
    print(f"\n{header}")
    print(f"Total candles: {len(data)}")
    print(f"Columns: {list(data.columns)}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    print(f"\n=== First 5 rows ({label}) ===")
    print(data.head())

    print(f"\n=== Last 5 rows ({label}) ===")
    print(data.tail())

    print(f"\n=== Basic Statistics ({label}) ===")
    print(data.describe())


# === Main Entry Point ===
if __name__ == "__main__":
    # Step 1: Create the Polygon client (uses API key from .env)
    client = get_polygon_client()

    # Step 2: Download both 5m and 1m candle data (up to 2 years of full trading days)
    # Data is saved to CSV as each interval finishes, so progress isn't lost.
    # If a chunk hits a 429 rate limit, it waits and retries automatically.
    print(f"Starting download for {TICKER}...")
    print(f"Lookback: {LOOKBACK_DAYS} days (~{LOOKBACK_DAYS // 365} years)")
    print(f"Retries: up to {MAX_RETRIES}x with {RETRY_WAIT_SECONDS}s wait on rate limit\n")

    all_data = download_all_intervals(client, TICKER)

    # Step 3: Inspect saved data
    print(f"\n{'=' * 60}")
    print(f"  Download complete! Inspecting saved data...")
    print(f"{'=' * 60}")

    for interval in INTERVAL_CONFIG.keys():
        loaded_data = load_from_csv(TICKER, interval)
        if not loaded_data.empty:
            inspect_data(loaded_data, label=interval)