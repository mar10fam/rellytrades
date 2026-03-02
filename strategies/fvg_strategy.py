"""
FVG Opening Range Breakout Strategy

This is Strategy #1: detect when price creates a Fair Value Gap (FVG)
that breaks through the opening range high or low.

How it works:
1. Mark the opening range — High and Low of the first 5-minute candle
   after market open (9:30 AM ET).
2. Switch to 1-minute candles starting at 9:35 AM (after the 5m candle closes).
3. Scan for Fair Value Gaps (FVGs) on the 1-minute chart.
4. Check if any FVG breaks through the opening range high (bullish) or low (bearish).
5. Return the first valid setup found for the day.

What is a Fair Value Gap (FVG)?
An FVG is a 3-candle pattern where candle 2 moved so aggressively that it left
a gap between candle 1's wick and candle 3's wick. This gap represents an
"imbalance" in price — an area where one side (buyers or sellers) overwhelmed
the other so quickly that no trading occurred in that zone.

- Bullish FVG: candle_3.Low > candle_1.High (gap above candle 1)
- Bearish FVG: candle_3.High < candle_1.Low (gap below candle 1)

Usage:
    from strategies.fvg_strategy import FVGStrategy

    strategy = FVGStrategy()
    setups = strategy.detect_setups(data_5m, data_1m)
    print(f"Found {len(setups)} valid setups")
"""

from dataclasses import dataclass
from datetime import date, time, datetime, timedelta
from typing import Optional

import pandas as pd

from strategies.base_strategy import BaseStrategy, Setup, Entry, Exit


# === FVG Data Class ===

@dataclass
class FVG:
    """
    Represents a single detected Fair Value Gap on the 1-minute chart.

    This is an internal data structure used during detection — it's not
    returned to the caller. Only FVGs that break through the opening range
    get promoted to a Setup.
    """
    direction: str          # "bullish" or "bearish"
    fvg_high: float         # Upper boundary of the gap zone
    fvg_low: float          # Lower boundary of the gap zone
    candle_1_high: float    # High of the first candle in the 3-candle pattern
    candle_1_low: float     # Low of the first candle in the 3-candle pattern
    candle_3_close: float   # Close of the third candle (used as entry price)
    candle_3_timestamp: datetime  # Timestamp of candle 3 (FVG completion time)


# === Constants ===

# Market open time in Eastern Time
MARKET_OPEN = time(9, 30)

# The 1-minute scan starts after the first 5-minute candle closes.
# The 9:30 AM 5-min candle covers 9:30:00 through 9:34:59,
# so 1-min scanning starts at 9:35 AM.
SCAN_START_TIME = time(9, 35)

# Default scan window end time — 2 hours after open = 11:30 AM ET
DEFAULT_SCAN_END_TIME = time(11, 30)


class FVGStrategy(BaseStrategy):
    """
    Opening Range + Fair Value Gap Breakout Strategy.

    Detects when price creates an FVG that breaks through the opening
    range high (bullish setup) or low (bearish setup) on the 1-minute chart.

    Args:
        scan_end_time: How late in the day to keep scanning for FVG breakouts.
                       Default is 11:30 AM ET (2 hours after open).
                       Pass a datetime.time object to customize.
    """

    def __init__(self, scan_end_time: time = DEFAULT_SCAN_END_TIME):
        self.scan_end_time = scan_end_time

    def detect_setup(
        self,
        data_5m: pd.DataFrame,
        data_1m: pd.DataFrame,
        trading_day: date,
    ) -> Optional[Setup]:
        """
        Analyze a single trading day for an FVG breakout through the opening range.

        Steps:
        1. Find the opening range from the 9:30 AM 5-minute candle.
        2. Scan 1-minute candles (9:35 AM to scan_end_time) for FVGs.
        3. Check if any FVG breaks through the opening range high or low.
        4. Return the first valid setup, or None.

        Args:
            data_5m: 5-minute candle data for this trading day (US/Eastern tz).
            data_1m: 1-minute candle data for this trading day (US/Eastern tz).
            trading_day: The date being analyzed.

        Returns:
            A Setup if a valid FVG breakout was found, otherwise None.
        """
        # Step 1: Get the opening range
        opening_range = self._get_opening_range(data_5m, trading_day)
        if opening_range is None:
            return None

        or_high, or_low = opening_range

        # Step 2: Get the 1-minute candles within the scan window
        scan_data = self._get_scan_window(data_1m, trading_day)
        if scan_data.empty or len(scan_data) < 3:
            # Need at least 3 candles to detect an FVG
            return None

        # Step 3: Find all FVGs in the scan window
        fvgs = self._find_fvgs(scan_data)
        if not fvgs:
            return None

        # Step 4: Find the first FVG that breaks through the opening range
        breakout_fvg = self._find_first_breakout(fvgs, or_high, or_low)
        if breakout_fvg is None:
            return None

        # Step 5: Find the first candle that closed outside the opening range
        # This candle's low (longs) or high (shorts) becomes the stop loss
        direction = "long" if breakout_fvg.direction == "bullish" else "short"
        breakout_candle = self._find_breakout_candle(
            scan_data, or_high, or_low, direction
        )

        # Default to the FVG candle 1 levels if no breakout candle found
        bo_low = breakout_candle["Low"] if breakout_candle is not None else breakout_fvg.candle_1_low
        bo_high = breakout_candle["High"] if breakout_candle is not None else breakout_fvg.candle_1_high

        # Step 6: Build and return the Setup
        return Setup(
            date=trading_day,
            direction=direction,
            entry_price=breakout_fvg.candle_3_close,
            fvg_high=breakout_fvg.fvg_high,
            fvg_low=breakout_fvg.fvg_low,
            opening_range_high=or_high,
            opening_range_low=or_low,
            timeframe_used="1m",
            fvg_timestamp=breakout_fvg.candle_3_timestamp,
            breakout_candle_low=float(bo_low),
            breakout_candle_high=float(bo_high),
        )

    def get_entry(self, setup: Setup, confirmation) -> Optional[Entry]:
        """
        Convert a detected setup into a trade entry.

        In Milestone 3 (no confirmation signals yet), every setup is
        entered directly — confirmation=None means "always enter."
        Milestone 4 will add confirmation gating here.

        Args:
            setup: A detected Setup from detect_setup().
            confirmation: A confirmation signal instance, or None.
                          When None, the setup is entered unconditionally.

        Returns:
            An Entry if the trade should be taken, or None to skip.
        """
        # M3: No confirmation required — enter every setup
        if confirmation is not None:
            # Future: ask the confirmation module if this setup is valid
            # For now, just enter anyway
            pass

        return Entry(
            date=setup.date,
            direction=setup.direction,
            price=setup.entry_price,
            timestamp=setup.fvg_timestamp,
        )

    def get_exit(self, entry: Entry, setup: Setup, risk_config: dict) -> Exit:
        """
        Calculate stop loss and take profit levels.

        Stop loss is placed at the first candle that closed outside the
        opening range:
        - Long trades: SL at that candle's Low (support from the breakout candle)
        - Short trades: SL at that candle's High (resistance from the breakout candle)

        This is tighter than using the full opening range, which means:
        - Smaller risk per trade
        - Take profit is closer and more reachable
        - Faster resolution (fewer EOD closes)

        Take profit is calculated from the risk distance times the R:R ratio:
        - Risk = distance from entry to stop loss
        - TP = entry + (risk * R:R ratio) for longs
        - TP = entry - (risk * R:R ratio) for shorts

        Args:
            entry: The confirmed trade entry.
            setup: The original Setup (contains breakout candle levels).
            risk_config: Must contain "risk_reward_ratio" (e.g., 2.0).

        Returns:
            An Exit with stop_loss, take_profit, and the R:R ratio used.
        """
        rr_ratio = risk_config.get("risk_reward_ratio", 2.0)

        if entry.direction == "long":
            # Long: SL at the breakout candle's low, TP above entry
            stop_loss = setup.breakout_candle_low
            risk = entry.price - stop_loss
            take_profit = entry.price + (risk * rr_ratio)
        else:
            # Short: SL at the breakout candle's high, TP below entry
            stop_loss = setup.breakout_candle_high
            risk = stop_loss - entry.price
            take_profit = entry.price - (risk * rr_ratio)

        return Exit(
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=rr_ratio,
        )

    # === Private Helper Methods ===

    def _get_opening_range(
        self,
        data_5m: pd.DataFrame,
        trading_day: date,
    ) -> Optional[tuple[float, float]]:
        """
        Find the opening range for a trading day.

        The opening range is defined as the High and Low of the first
        5-minute candle at 9:30 AM ET.

        Args:
            data_5m: 5-minute candle data for this day (US/Eastern tz).
            trading_day: The date to find the opening range for.

        Returns:
            A tuple of (or_high, or_low), or None if no 9:30 candle exists.
        """
        # Build the exact timestamp for 9:30 AM ET on this day
        target_time = MARKET_OPEN

        # Filter to candles at exactly 9:30 AM
        # We use .indexer_at_time() or simple time comparison
        candles_at_open = data_5m[
            data_5m.index.time == target_time
        ]

        if candles_at_open.empty:
            # No 9:30 candle — could be a holiday or data gap
            return None

        # Take the first (and should be only) 9:30 candle
        opening_candle = candles_at_open.iloc[0]

        or_high = float(opening_candle["High"])
        or_low = float(opening_candle["Low"])

        return (or_high, or_low)

    def _get_scan_window(
        self,
        data_1m: pd.DataFrame,
        trading_day: date,
    ) -> pd.DataFrame:
        """
        Extract the 1-minute candles within the scan window for FVG detection.

        The scan window starts at 9:35 AM ET (after the 5-min opening range
        candle closes) and ends at self.scan_end_time (default 11:30 AM ET).

        Args:
            data_1m: 1-minute candle data for this day (US/Eastern tz).
            trading_day: The date being analyzed.

        Returns:
            A DataFrame slice containing only candles within the scan window.
        """
        # Filter by time of day — only keep candles between scan start and end
        candle_times = data_1m.index.time

        # Include candles from SCAN_START_TIME up to (but not after) scan_end_time
        mask = (candle_times >= SCAN_START_TIME) & (candle_times <= self.scan_end_time)

        return data_1m[mask]

    def _find_breakout_candle(
        self,
        scan_data: pd.DataFrame,
        or_high: float,
        or_low: float,
        direction: str,
    ) -> Optional[pd.Series]:
        """
        Find the first 1-minute candle that closed outside the opening range.

        For longs: the first candle whose Close > OR high.
        For shorts: the first candle whose Close < OR low.

        This candle's Low (longs) or High (shorts) is used as the stop loss,
        since it represents the nearest support/resistance from the actual
        breakout move.

        Args:
            scan_data: 1-minute candle data within the scan window.
            or_high: Opening range high price.
            or_low: Opening range low price.
            direction: "long" or "short".

        Returns:
            The first candle (as a pandas Series) that closed outside the OR,
            or None if no such candle exists.
        """
        if direction == "long":
            # First candle that closed above the opening range high
            breakout_candles = scan_data[scan_data["Close"] > or_high]
        else:
            # First candle that closed below the opening range low
            breakout_candles = scan_data[scan_data["Close"] < or_low]

        if breakout_candles.empty:
            return None

        return breakout_candles.iloc[0]

    def _find_fvgs(self, scan_data: pd.DataFrame) -> list[FVG]:
        """
        Scan 1-minute candles for Fair Value Gaps (FVGs).

        An FVG is a 3-candle pattern where candle 2 moved so aggressively
        that it left a gap between candle 1's wick and candle 3's wick:

        Bullish FVG: candle_3.Low > candle_1.High
          - The gap zone is from candle_1.High (bottom) to candle_3.Low (top)
          - Means buyers pushed price up so fast that the gap was never traded

        Bearish FVG: candle_3.High < candle_1.Low
          - The gap zone is from candle_3.High (bottom) to candle_1.Low (top)
          - Means sellers pushed price down so fast that the gap was never traded

        We use consecutive ROWS (not consecutive clock minutes) because
        some minutes may be missing if no trades occurred.

        Args:
            scan_data: 1-minute candle data within the scan window.

        Returns:
            A list of FVG objects found in the data.
        """
        fvgs = []

        # Slide a 3-candle window across the data
        # i is the index of candle 1, i+1 is candle 2, i+2 is candle 3
        for i in range(len(scan_data) - 2):
            candle_1 = scan_data.iloc[i]
            # candle_2 is scan_data.iloc[i + 1] — we don't need its values directly,
            # but it's the "aggressive" candle that creates the gap
            candle_3 = scan_data.iloc[i + 2]

            candle_1_high = float(candle_1["High"])
            candle_1_low = float(candle_1["Low"])
            candle_3_high = float(candle_3["High"])
            candle_3_low = float(candle_3["Low"])
            candle_3_close = float(candle_3["Close"])
            candle_3_timestamp = scan_data.index[i + 2]

            # Check for Bullish FVG: candle 3's low is above candle 1's high
            if candle_3_low > candle_1_high:
                fvgs.append(FVG(
                    direction="bullish",
                    fvg_high=candle_3_low,    # Top of the gap
                    fvg_low=candle_1_high,    # Bottom of the gap
                    candle_1_high=candle_1_high,
                    candle_1_low=candle_1_low,
                    candle_3_close=candle_3_close,
                    candle_3_timestamp=candle_3_timestamp,
                ))

            # Check for Bearish FVG: candle 3's high is below candle 1's low
            elif candle_3_high < candle_1_low:
                fvgs.append(FVG(
                    direction="bearish",
                    fvg_high=candle_1_low,    # Top of the gap
                    fvg_low=candle_3_high,    # Bottom of the gap
                    candle_1_high=candle_1_high,
                    candle_1_low=candle_1_low,
                    candle_3_close=candle_3_close,
                    candle_3_timestamp=candle_3_timestamp,
                ))

        return fvgs

    def _find_first_breakout(
        self,
        fvgs: list[FVG],
        or_high: float,
        or_low: float,
    ) -> Optional[FVG]:
        """
        Find the first FVG that breaks through the opening range.

        An FVG "breaks through" the opening range when:

        Bullish breakout (through OR high):
          - The FVG is bullish (price surging upward)
          - Candle 1 was at or below the opening range high
            (price hadn't broken out yet)
          - Candle 3's low (the bottom of the gap) is above the OR high
            (price gapped THROUGH the level, not just touched it)

        Bearish breakout (through OR low):
          - The FVG is bearish (price plunging downward)
          - Candle 1 was at or above the opening range low
            (price hadn't broken down yet)
          - Candle 3's high (the top of the gap) is below the OR low
            (price gapped THROUGH the level)

        Args:
            fvgs: List of FVGs detected in the scan window.
            or_high: Opening range high price.
            or_low: Opening range low price.

        Returns:
            The first FVG that qualifies as a breakout, or None.
        """
        for fvg in fvgs:
            if fvg.direction == "bullish":
                # Bullish breakout: price gaps up through the OR high
                # Candle 1 was still at or below the OR high level
                candle_1_was_below_or_high = fvg.candle_1_high <= or_high
                # The gap starts above the OR high (candle 3's low > OR high)
                gap_is_above_or_high = fvg.fvg_low >= or_high
                # Alternative: the gap spans the OR high (fvg_low <= or_high <= fvg_high)
                gap_spans_or_high = fvg.fvg_low <= or_high <= fvg.fvg_high

                if candle_1_was_below_or_high and (gap_is_above_or_high or gap_spans_or_high):
                    return fvg

            elif fvg.direction == "bearish":
                # Bearish breakout: price gaps down through the OR low
                # Candle 1 was still at or above the OR low level
                candle_1_was_above_or_low = fvg.candle_1_low >= or_low
                # The gap ends below the OR low (candle 3's high < OR low)
                gap_is_below_or_low = fvg.fvg_high <= or_low
                # Alternative: the gap spans the OR low (fvg_low <= or_low <= fvg_high)
                gap_spans_or_low = fvg.fvg_low <= or_low <= fvg.fvg_high

                if candle_1_was_above_or_low and (gap_is_below_or_low or gap_spans_or_low):
                    return fvg

        return None
