"""
VWAP Cross confirmation signal.

Checks whether the entry price is on the correct side of the Volume
Weighted Average Price (VWAP). For long trades, price should be above
VWAP (bullish bias). For short trades, price should be below VWAP
(bearish bias).

VWAP is calculated from the start of the trading day using cumulative
(price × volume) / cumulative volume.

Usage:
    from confirmations.vwap_cross import VWAPCross

    vc = VWAPCross()
"""

import numpy as np
import pandas as pd

from confirmations.base_confirmation import BaseConfirmation
from strategies.base_strategy import Setup


class VWAPCross(BaseConfirmation):
    """
    Confirms a setup if the entry price is on the right side of VWAP.

    - Long trades: entry price must be ABOVE VWAP (bullish momentum)
    - Short trades: entry price must be BELOW VWAP (bearish momentum)

    VWAP is computed intraday from market open using the typical price
    (High + Low + Close) / 3, weighted by volume.
    """

    def check(self, setup: Setup, day_1m: pd.DataFrame) -> bool:
        """
        Check if the entry price is on the correct side of VWAP.

        Computes VWAP from the day's 1-minute data up to the entry time,
        then checks whether the entry price confirms the trade direction.

        Args:
            setup: The detected Setup (uses fvg_timestamp, entry_price, direction).
            day_1m: Full 1-minute candle data for the trading day.

        Returns:
            True if entry price is above VWAP (long) or below VWAP (short).
        """
        if day_1m.empty:
            return False

        # Get candles up to and including the entry timestamp
        candles_up_to_entry = day_1m[day_1m.index <= setup.fvg_timestamp]

        if candles_up_to_entry.empty:
            return False

        # Calculate VWAP: cumulative(typical_price × volume) / cumulative(volume)
        # Typical price = (High + Low + Close) / 3
        typical_price = (
            candles_up_to_entry["High"]
            + candles_up_to_entry["Low"]
            + candles_up_to_entry["Close"]
        ) / 3.0

        volume = candles_up_to_entry["Volume"]

        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()

        # Avoid division by zero
        if float(cumulative_vol.iloc[-1]) == 0:
            return False

        # VWAP at the entry time is the last value in the cumulative series
        vwap = float(cumulative_tp_vol.iloc[-1] / cumulative_vol.iloc[-1])

        # Check direction alignment
        if setup.direction == "long":
            return setup.entry_price > vwap
        else:
            return setup.entry_price < vwap
