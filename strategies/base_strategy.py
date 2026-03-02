"""
Base strategy interface that all trading strategies must implement.

This module defines:
- The abstract base class (BaseStrategy) with the required method signatures
- Data classes for Setup, Entry, and Exit that strategies return

Any new strategy just needs to subclass BaseStrategy, implement
detect_setup(), get_entry(), and get_exit(), and it will work
with the backtesting engine automatically.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import pandas as pd


# === Data Classes ===
# These define the shape of data returned by strategy methods.
# Using dataclasses for clear, readable structure with type hints.

@dataclass
class Setup:
    """
    Represents a detected trade setup — a potential trade opportunity
    found by a strategy's detection logic.

    This is "detection only" — it doesn't mean a trade was taken.
    The backtesting engine (Milestone 3) will decide whether to
    act on a setup based on confirmation signals and risk rules.
    """
    date: date                    # The trading day this setup was found on
    direction: str                # "long" or "short"
    entry_price: float            # Price at which one would enter (close of FVG candle 3)
    fvg_high: float               # Upper boundary of the Fair Value Gap zone
    fvg_low: float                # Lower boundary of the Fair Value Gap zone
    opening_range_high: float     # High of the opening range candle
    opening_range_low: float      # Low of the opening range candle
    timeframe_used: str           # Which timeframe detected this (e.g., "1m")
    fvg_timestamp: datetime       # Timestamp of the FVG completion (candle 3)
    breakout_candle_low: float    # Low of the first candle that closed outside the OR
    breakout_candle_high: float   # High of the first candle that closed outside the OR


@dataclass
class Entry:
    """
    Represents a confirmed trade entry — returned by get_entry()
    after a setup passes confirmation checks.

    Placeholder for Milestone 3 — not used in Milestone 2.
    """
    date: date
    direction: str                # "long" or "short"
    price: float                  # Actual entry price
    timestamp: datetime           # When the entry was triggered


@dataclass
class Exit:
    """
    Represents trade exit levels — stop loss and take profit.

    Placeholder for Milestone 3 — not used in Milestone 2.
    """
    stop_loss: float              # Price level to exit at a loss
    take_profit: float            # Price level to exit at a profit
    risk_reward_ratio: float      # The R:R ratio used to calculate these levels


# === Base Strategy Class ===

class BaseStrategy(ABC):
    """
    Abstract base class that all trading strategies must inherit from.

    To create a new strategy:
    1. Create a new file in strategies/ (e.g., my_strategy.py)
    2. Subclass BaseStrategy
    3. Implement detect_setup(), get_entry(), and get_exit()

    The backtesting engine only interacts with strategies through
    this interface — it never imports or knows about specific strategies.

    Example:
        class MyStrategy(BaseStrategy):
            def detect_setup(self, data_5m, data_1m, trading_day):
                # Your detection logic here
                return Setup(...) or None
    """

    @abstractmethod
    def detect_setup(
        self,
        data_5m: pd.DataFrame,
        data_1m: pd.DataFrame,
        trading_day: date,
    ) -> Optional[Setup]:
        """
        Analyze a single trading day and return a Setup if conditions are met.

        This is the core detection method. Each strategy defines what
        constitutes a valid setup (e.g., FVG breakout through opening range).

        Args:
            data_5m: 5-minute candle data for the trading day.
                     Columns: Open, High, Low, Close, Volume.
                     Index: DatetimeIndex in US/Eastern timezone.
            data_1m: 1-minute candle data for the trading day.
                     Same structure as data_5m but at 1-min resolution.
            trading_day: The date to analyze.

        Returns:
            A Setup object if a valid setup was detected, or None if not.
        """
        pass

    @abstractmethod
    def get_entry(self, setup: Setup, confirmation) -> Optional[Entry]:
        """
        Determine whether to enter a trade based on a detected setup
        and a confirmation signal.

        Placeholder for Milestone 3+ — not used in Milestone 2.

        Args:
            setup: A previously detected Setup from detect_setup().
            confirmation: A confirmation signal module instance
                          (e.g., VolumeSpike, VWAPCross).

        Returns:
            An Entry object if confirmed, or None if the confirmation
            signal did not trigger.
        """
        pass

    @abstractmethod
    def get_exit(self, entry: Entry, setup: Setup, risk_config: dict) -> Exit:
        """
        Calculate stop loss and take profit levels for a confirmed entry.

        Each strategy defines its own SL placement logic based on its
        setup details. The engine passes both the entry and the original
        setup so the strategy can reference whatever it needs (e.g.,
        opening range levels, FVG zone, etc.) without the engine
        needing to understand those concepts.

        Args:
            entry: A confirmed Entry from get_entry().
            setup: The original Setup that generated this entry.
            risk_config: Dictionary with risk parameters like
                         risk_reward_ratio, risk_per_trade, etc.

        Returns:
            An Exit object with stop loss and take profit levels.
        """
        pass

    def detect_setups(
        self,
        data_5m: pd.DataFrame,
        data_1m: pd.DataFrame,
    ) -> list[Setup]:
        """
        Batch method: scan ALL trading days and return every detected setup.

        This is a concrete method (not abstract) — it works the same for
        every strategy. It finds all unique trading days in the data,
        then calls detect_setup() for each day.

        Args:
            data_5m: 5-minute candle data across many days.
                     Index: DatetimeIndex in US/Eastern timezone.
            data_1m: 1-minute candle data across many days.
                     Same structure as data_5m.

        Returns:
            A list of Setup objects — one per day that had a valid setup.
            Days with no valid setup are silently skipped.
        """
        # Ensure timezone is US/Eastern for consistent time filtering
        data_5m = _ensure_eastern_timezone(data_5m)
        data_1m = _ensure_eastern_timezone(data_1m)

        # Get unique trading days from the 5m data
        # .normalize() strips the time component, leaving just the date
        trading_days = data_5m.index.normalize().unique()

        setups = []
        for day_timestamp in trading_days:
            trading_day = day_timestamp.date()

            # Slice both DataFrames to just this trading day
            day_str = trading_day.isoformat()
            day_5m = data_5m.loc[day_str]
            day_1m = data_1m.loc[day_str] if day_str in data_1m.index else pd.DataFrame()

            # Skip days where we don't have 1m data
            if day_1m.empty:
                continue

            # Ask the strategy to detect a setup for this day
            setup = self.detect_setup(day_5m, day_1m, trading_day)

            if setup is not None:
                setups.append(setup)

        print(f"Scanned {len(trading_days)} trading days, found {len(setups)} valid setups")
        return setups


# === Helper Functions ===

def _ensure_eastern_timezone(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the DataFrame's DatetimeIndex is in US/Eastern timezone.

    The CSV files contain mixed UTC offsets (-05:00 for EST, -04:00 for EDT).
    When pandas reads these, it converts them to UTC internally. This function
    converts back to US/Eastern so we can filter by local time (e.g., 9:30 AM).

    Args:
        data: DataFrame with a DatetimeIndex (possibly in UTC or with offset)

    Returns:
        The same DataFrame with its index converted to US/Eastern timezone
    """
    if data.index.tz is None:
        # No timezone info — assume it's already Eastern (shouldn't happen with our CSVs)
        data.index = data.index.tz_localize("US/Eastern")
    else:
        # Has timezone info — convert to Eastern
        data.index = data.index.tz_convert("US/Eastern")

    return data
