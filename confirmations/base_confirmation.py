"""
Base interface for all confirmation signals.

Every confirmation module must inherit from BaseConfirmation and implement
the check() method. Confirmations are strategy-agnostic — they can be
reused across any strategy that passes a Setup and 1-minute data.

The engine passes confirmation instances to strategy.get_entry(), which
calls check() to decide whether the setup should be entered.
"""

from abc import ABC, abstractmethod

import pandas as pd

from strategies.base_strategy import Setup


class BaseConfirmation(ABC):
    """
    Abstract base class for confirmation signals.

    Each confirmation answers one question: "Should this setup be entered?"
    It receives the setup details and the full day's 1-minute data so it
    can compute any technical indicator it needs.

    To create a new confirmation:
    1. Create a new file in confirmations/
    2. Subclass BaseConfirmation
    3. Implement check() — return True to confirm, False to reject

    Example:
        class MyConfirmation(BaseConfirmation):
            def check(self, setup, day_1m):
                # Your logic here
                return True or False
    """

    @abstractmethod
    def check(self, setup: Setup, day_1m: pd.DataFrame) -> bool:
        """
        Evaluate whether a detected setup passes this confirmation filter.

        Args:
            setup: A detected Setup from the strategy's detect_setup().
                   Contains entry_price, direction, fvg_timestamp, etc.
            day_1m: Full 1-minute candle data for the trading day.
                    Columns: Open, High, Low, Close, Volume.
                    Index: DatetimeIndex in US/Eastern timezone.

        Returns:
            True if the setup passes this confirmation, False to skip it.
        """
        pass
