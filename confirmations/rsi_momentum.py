"""
RSI Momentum confirmation signal.

Checks whether the Relative Strength Index (RSI) supports the trade
direction. For long trades, RSI should be above a threshold (default 50),
indicating bullish momentum. For short trades, RSI should be below
the threshold, indicating bearish momentum.

RSI is calculated using the standard Wilder smoothing method over a
configurable period (default 14 candles).

Usage:
    from confirmations.rsi_momentum import RSIMomentum

    # Default: RSI period=14, threshold=50
    rsi = RSIMomentum()

    # Custom: RSI period=10, threshold=45
    rsi = RSIMomentum(period=10, threshold=45)
"""

import numpy as np
import pandas as pd

from confirmations.base_confirmation import BaseConfirmation
from strategies.base_strategy import Setup


class RSIMomentum(BaseConfirmation):
    """
    Confirms a setup if RSI supports the trade direction.

    - Long trades: RSI must be ABOVE the threshold (bullish momentum)
    - Short trades: RSI must be BELOW the threshold (bearish momentum)

    Uses the standard Wilder smoothing RSI calculation.

    Args:
        period: Number of candles for the RSI calculation (default 14).
        threshold: The RSI level to check against (default 50).
                   Longs need RSI > threshold, shorts need RSI < threshold.
    """

    def __init__(self, period: int = 14, threshold: float = 50.0):
        self.period = period
        self.threshold = threshold

    def check(self, setup: Setup, day_1m: pd.DataFrame) -> bool:
        """
        Check if RSI confirms the trade direction.

        Computes RSI on the 1-minute close prices up to the entry
        timestamp. If there aren't enough candles for the RSI period,
        the confirmation fails (returns False).

        Args:
            setup: The detected Setup (uses fvg_timestamp, direction).
            day_1m: Full 1-minute candle data for the trading day.

        Returns:
            True if RSI supports the trade direction, False otherwise.
        """
        if day_1m.empty:
            return False

        # Get candles up to and including the entry timestamp
        candles_up_to_entry = day_1m[day_1m.index <= setup.fvg_timestamp]

        # Need at least period + 1 candles to calculate RSI
        if len(candles_up_to_entry) < self.period + 1:
            return False

        # Calculate RSI using Wilder smoothing
        closes = candles_up_to_entry["Close"].astype(float)
        rsi_value = self._calculate_rsi(closes)

        if rsi_value is None:
            return False

        # Check direction alignment
        if setup.direction == "long":
            return rsi_value > self.threshold
        else:
            return rsi_value < self.threshold

    def _calculate_rsi(self, closes: pd.Series) -> float | None:
        """
        Calculate RSI using Wilder's smoothing method.

        This is the standard RSI calculation:
        1. Compute price changes (deltas)
        2. Separate gains and losses
        3. Use exponential moving average (Wilder smoothing) for avg gain/loss
        4. RS = avg_gain / avg_loss
        5. RSI = 100 - (100 / (1 + RS))

        Args:
            closes: Series of close prices ordered chronologically.

        Returns:
            The RSI value at the most recent candle, or None if it
            can't be calculated.
        """
        # Price changes between consecutive candles
        delta = closes.diff()

        # Separate gains (positive changes) and losses (negative changes)
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)

        # Wilder smoothing: EMA with alpha = 1/period
        # First value is the simple average of the first N periods
        avg_gain = gains.ewm(alpha=1.0 / self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1.0 / self.period, min_periods=self.period, adjust=False).mean()

        # Get the most recent values
        last_avg_gain = float(avg_gain.iloc[-1])
        last_avg_loss = float(avg_loss.iloc[-1])

        # Handle edge case: no losses means RSI = 100
        if last_avg_loss == 0:
            return 100.0

        rs = last_avg_gain / last_avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi
