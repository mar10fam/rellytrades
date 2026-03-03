"""
Composite confirmation — stacks multiple confirmations with AND logic.

When multiple confirmation signals are stacked, CompositeConfirmation
requires ALL of them to pass before a setup is entered. This is the
"support stacking" model: each confirmation is a filter that must agree.

Usage:
    from confirmations.composite import CompositeConfirmation
    from confirmations.volume_spike import VolumeSpike
    from confirmations.vwap_cross import VWAPCross

    combo = CompositeConfirmation([VolumeSpike(), VWAPCross()])
    # combo.check(setup, day_1m) returns True only if BOTH pass
"""

import pandas as pd

from confirmations.base_confirmation import BaseConfirmation
from strategies.base_strategy import Setup


class CompositeConfirmation(BaseConfirmation):
    """
    Combines multiple confirmations with AND logic.

    All inner confirmations must return True for the composite to pass.
    If any single confirmation rejects the setup, the composite rejects it.

    Args:
        confirmations: A list of BaseConfirmation instances to stack.
    """

    def __init__(self, confirmations: list[BaseConfirmation]):
        self.confirmations = confirmations

    def check(self, setup: Setup, day_1m: pd.DataFrame) -> bool:
        """
        Check all stacked confirmations — returns True only if ALL pass.

        Short-circuits on the first failure for efficiency.

        Args:
            setup: The detected Setup to evaluate.
            day_1m: Full 1-minute candle data for the trading day.

        Returns:
            True if every confirmation passes, False otherwise.
        """
        for confirmation in self.confirmations:
            if not confirmation.check(setup, day_1m):
                return False
        return True
