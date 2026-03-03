"""
Volume Spike confirmation signal.

Checks whether the volume on the entry candle (or the candle at the
FVG timestamp) is significantly higher than the recent average. A volume
spike suggests strong conviction behind the breakout, making the trade
more likely to follow through.

Usage:
    from confirmations.volume_spike import VolumeSpike

    # Default: 2x average volume over 20-candle lookback
    vs = VolumeSpike()

    # Custom: 1.5x average volume over 10-candle lookback
    vs = VolumeSpike(multiplier=1.5, lookback=10)
"""

import pandas as pd

from confirmations.base_confirmation import BaseConfirmation
from strategies.base_strategy import Setup


class VolumeSpike(BaseConfirmation):
    """
    Confirms a setup if the entry candle's volume is above a threshold
    relative to the recent rolling average volume.

    A volume spike indicates that many participants are behind the move,
    increasing the chance of follow-through.

    Args:
        multiplier: How many times the average volume the entry candle
                    must exceed (default 2.0 = 2x average).
        lookback: Number of candles to use for the rolling average
                  (default 20).
    """

    def __init__(self, multiplier: float = 2.0, lookback: int = 20):
        self.multiplier = multiplier
        self.lookback = lookback

    def check(self, setup: Setup, day_1m: pd.DataFrame) -> bool:
        """
        Check if the entry candle has a volume spike.

        Finds the candle at or just before the FVG timestamp and compares
        its volume to the rolling average of the preceding candles.

        Args:
            setup: The detected Setup (uses fvg_timestamp for the entry candle).
            day_1m: Full 1-minute candle data for the trading day.

        Returns:
            True if entry candle volume >= multiplier × rolling avg volume.
        """
        if day_1m.empty or "Volume" not in day_1m.columns:
            return False

        # Find candles up to and including the entry timestamp
        candles_up_to_entry = day_1m[day_1m.index <= setup.fvg_timestamp]

        if len(candles_up_to_entry) < 2:
            # Not enough data to compute an average
            return False

        # The entry candle is the last one in the slice
        entry_candle_volume = float(candles_up_to_entry.iloc[-1]["Volume"])

        # Calculate rolling average volume from the candles BEFORE the entry candle
        preceding_candles = candles_up_to_entry.iloc[:-1]

        # Use the last N candles (lookback) for the average
        lookback_candles = preceding_candles.tail(self.lookback)

        if lookback_candles.empty:
            return False

        avg_volume = float(lookback_candles["Volume"].mean())

        if avg_volume == 0:
            return False

        # Check if the entry candle's volume exceeds the threshold
        return entry_candle_volume >= (avg_volume * self.multiplier)
