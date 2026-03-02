"""
Backtesting Engine — simulates trades from any strategy's detected setups.

This module is completely strategy-agnostic. It:
1. Loads historical data for a ticker
2. Asks the strategy to detect setups
3. Converts setups to entries and exits via the strategy interface
4. Simulates each trade candle-by-candle on 1-minute data
5. Records results in a trade log

The engine never imports or references any specific strategy — it only
interacts through the BaseStrategy interface (Setup, Entry, Exit).

Usage:
    from engine.backtester import Backtester
    from strategies.fvg_strategy import FVGStrategy

    bt = Backtester(strategy=FVGStrategy())
    results = bt.run('AAPL', start='2025-01-01', end='2025-01-31')
    print(results.total_pnl)
    print(results.win_rate)
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Optional

import pandas as pd

from strategies.base_strategy import BaseStrategy, Setup, Entry, Exit


# === Default Configuration ===

DEFAULT_RISK_CONFIG = {
    "risk_reward_ratio": 2.0,  # Take profit at 2x the risk distance
    "starting_balance": 1000.0,  # Initial account balance in dollars
    "risk_per_trade": 0.01,  # Risk 1% of current balance per trade
}

# Market close time — positions are force-closed at end of day
MARKET_CLOSE_TIME = time(16, 0)


# === Data Classes ===

@dataclass
class TradeRecord:
    """
    A single completed trade — records everything about the entry, exit, and result.

    Created by the engine after simulating a trade candle-by-candle.
    This is what goes into the trade log.
    """
    entry_time: datetime       # Timestamp when the trade was entered
    exit_time: datetime        # Timestamp when the trade was exited
    ticker: str                # Which stock this trade is on
    direction: str             # "long" or "short"
    entry_price: float         # Price at which the trade was entered
    exit_price: float          # Price at which the trade was exited
    stop_loss: float           # The stop loss level
    take_profit: float         # The take profit level
    pnl: float                 # Dollar profit/loss on this trade (per share)
    result: str                # "win", "loss", or "closed_eod"
    or_high: float             # Opening range high for this trade's day
    or_low: float              # Opening range low for this trade's day
    shares: float              # Number of shares traded (fractional allowed)
    dollar_pnl: float          # Actual dollar P&L for this trade (pnl × shares)
    balance_after: float       # Account balance after this trade


@dataclass
class BacktestResult:
    """
    The complete result of a backtest run — contains all trades and summary stats.

    Computed properties (win_rate, total_pnl, etc.) are calculated from
    the trade list so they're always consistent.
    """
    ticker: str                          # Which stock was backtested
    starting_balance: float = 1000.0     # Initial account balance
    risk_per_trade: float = 0.01         # Risk percentage per trade
    start_date: Optional[str] = None     # Start of the date range tested
    end_date: Optional[str] = None       # End of the date range tested
    trades: list[TradeRecord] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        """Total number of trades executed."""
        return len(self.trades)

    @property
    def wins(self) -> int:
        """Number of winning trades (take profit hit)."""
        return sum(1 for t in self.trades if t.result == "win")

    @property
    def losses(self) -> int:
        """Number of losing trades (stop loss hit)."""
        return sum(1 for t in self.trades if t.result == "loss")

    @property
    def eod_closes(self) -> int:
        """Number of trades closed at end of day (neither SL nor TP hit)."""
        return sum(1 for t in self.trades if t.result == "closed_eod")

    @property
    def win_rate(self) -> float:
        """Win rate as a decimal (0.0 to 1.0). Returns 0 if no trades."""
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def total_pnl(self) -> float:
        """Total dollar P&L across all trades."""
        return sum(t.dollar_pnl for t in self.trades)

    @property
    def ending_balance(self) -> float:
        """Account balance after all trades."""
        if not self.trades:
            return self.starting_balance
        return self.trades[-1].balance_after

    @property
    def total_return(self) -> float:
        """Total return as a decimal (e.g., 0.05 = 5% gain)."""
        if self.starting_balance == 0:
            return 0.0
        return (self.ending_balance - self.starting_balance) / self.starting_balance

    @property
    def avg_pnl(self) -> float:
        """Average dollar P&L per trade. Returns 0 if no trades."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

    @property
    def avg_win(self) -> float:
        """Average dollar P&L on winning trades. Returns 0 if no wins."""
        winning_trades = [t for t in self.trades if t.result == "win"]
        if not winning_trades:
            return 0.0
        return sum(t.dollar_pnl for t in winning_trades) / len(winning_trades)

    @property
    def avg_loss(self) -> float:
        """Average dollar P&L on losing trades. Returns 0 if no losses."""
        losing_trades = [t for t in self.trades if t.result == "loss"]
        if not losing_trades:
            return 0.0
        return sum(t.dollar_pnl for t in losing_trades) / len(losing_trades)

    @property
    def profit_factor(self) -> float:
        """
        Ratio of gross profits to gross losses.
        > 1.0 means the strategy is profitable overall.
        Returns 0 if no losing trades (can't divide by zero).
        """
        gross_profit = sum(t.dollar_pnl for t in self.trades if t.dollar_pnl > 0)
        gross_loss = abs(sum(t.dollar_pnl for t in self.trades if t.dollar_pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def print_summary(self) -> None:
        """Print a clean summary of the backtest results to the terminal."""
        print(f"\n{'=' * 60}")
        print(f"  Backtest Results: {self.ticker}")
        if self.start_date and self.end_date:
            print(f"  Period: {self.start_date} to {self.end_date}")
        print(f"{'=' * 60}")
        print(f"  Total trades:   {self.total_trades}")
        print(f"  Wins:           {self.wins}")
        print(f"  Losses:         {self.losses}")
        print(f"  EOD closes:     {self.eod_closes}")
        print(f"  Win rate:       {self.win_rate:.1%}")
        print(f"  Risk per trade: {self.risk_per_trade:.1%}")
        print(f"  Starting bal:   ${self.starting_balance:,.2f}")
        print(f"  Ending bal:     ${self.ending_balance:,.2f}")
        print(f"  Total P&L:      ${self.total_pnl:+,.2f}")
        print(f"  Total return:   {self.total_return:+.2%}")
        print(f"  Avg P&L/trade:  ${self.avg_pnl:+.2f}")
        print(f"  Avg win:        ${self.avg_win:+.2f}")
        print(f"  Avg loss:       ${self.avg_loss:+.2f}")
        print(f"  Profit factor:  {self.profit_factor:.2f}")
        print(f"{'=' * 60}")

    def print_trades(self, max_trades: int = 50) -> None:
        """
        Print the trade log in a readable table format.

        Args:
            max_trades: Maximum number of trades to print (to avoid flooding the terminal).
        """
        if not self.trades:
            print("  No trades to display.")
            return

        print(f"\n{'=' * 160}")
        print(f"  Trade Log ({min(len(self.trades), max_trades)} of {len(self.trades)} trades)")
        print(f"{'=' * 160}")
        print(
            f"  {'#':>3s}  {'Date':10s}  {'Ticker':6s}  {'Dir':5s}  "
            f"{'OR High':>8s}  {'OR Low':>8s}  "
            f"{'Entry':>8s}  {'Exit':>8s}  {'SL':>8s}  {'TP':>8s}  "
            f"{'Shares':>8s}  {'$ P&L':>9s}  {'Balance':>10s}  "
            f"{'Result':10s}  {'Exit Time':5s}"
        )
        print(f"  {'-' * 154}")

        for i, trade in enumerate(self.trades[:max_trades], 1):
            print(
                f"  {i:3d}  {trade.entry_time.strftime('%Y-%m-%d'):10s}  "
                f"{trade.ticker:6s}  "
                f"{trade.direction:5s}  "
                f"${trade.or_high:>7.2f}  ${trade.or_low:>7.2f}  "
                f"${trade.entry_price:>7.2f}  ${trade.exit_price:>7.2f}  "
                f"${trade.stop_loss:>7.2f}  ${trade.take_profit:>7.2f}  "
                f"{trade.shares:>8.2f}  ${trade.dollar_pnl:>+8.2f}  "
                f"${trade.balance_after:>9.2f}  "
                f"{trade.result:10s}  "
                f"{trade.exit_time.strftime('%H:%M'):5s}"
            )

        if len(self.trades) > max_trades:
            print(f"  ... and {len(self.trades) - max_trades} more trades")

        print(f"  {'-' * 154}")


# === Backtester Class ===

class Backtester:
    """
    Strategy-agnostic backtesting engine.

    Takes any strategy that follows the BaseStrategy interface, runs it
    against historical data, and simulates trades candle-by-candle.

    The engine never knows what strategy it's running — it only calls
    detect_setups(), get_entry(), and get_exit() through the interface.

    Args:
        strategy: Any class that inherits from BaseStrategy.
        risk_config: Dictionary with risk parameters. For Milestone 3,
                     only "risk_reward_ratio" is used (default 3.0).
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        risk_config: Optional[dict] = None,
    ):
        self.strategy = strategy
        self.risk_config = risk_config or DEFAULT_RISK_CONFIG.copy()

    def run(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run a complete backtest for a ticker over a date range.

        Steps:
        1. Load 5m and 1m CSV data for the ticker.
        2. Filter to the requested date range (if provided).
        3. Detect setups via the strategy.
        4. For each setup, get entry and exit levels from the strategy.
        5. Simulate each trade candle-by-candle on 1m data.
        6. Return a BacktestResult with all trades and summary stats.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL", "AMD").
            start: Start date in "YYYY-MM-DD" format (inclusive). Optional.
            end: End date in "YYYY-MM-DD" format (inclusive). Optional.

        Returns:
            A BacktestResult with the full trade log and computed metrics.
        """
        # Import load_from_csv from main — keeps the engine from needing
        # to know about file paths or CSV formats
        from main import load_from_csv

        # Step 1: Load data
        data_5m = load_from_csv(ticker, "5m")
        data_1m = load_from_csv(ticker, "1m")

        if data_5m.empty or data_1m.empty:
            print(f"ERROR: Missing data for {ticker}. Run download first.")
            return BacktestResult(ticker=ticker, start_date=start, end_date=end)

        # Step 2: Filter to date range (if provided)
        if start:
            data_5m = data_5m.loc[start:]
            data_1m = data_1m.loc[start:]
        if end:
            data_5m = data_5m.loc[:end]
            data_1m = data_1m.loc[:end]

        # Read account simulation config
        starting_balance = self.risk_config.get("starting_balance", 1000.0)
        risk_per_trade = self.risk_config.get("risk_per_trade", 0.01)
        current_balance = starting_balance

        # Step 3: Detect setups
        print(f"\n{'=' * 60}")
        print(f"  Running backtest on {ticker}")
        if start and end:
            print(f"  Period: {start} to {end}")
        print(f"  Starting balance: ${starting_balance:,.2f}")
        print(f"  Risk per trade:   {risk_per_trade:.1%}")
        print(f"  R:R ratio:        {self.risk_config.get('risk_reward_ratio', 2.0)}")
        print(f"{'=' * 60}")

        setups = self.strategy.detect_setups(data_5m, data_1m)

        if not setups:
            print("No setups found — nothing to backtest.")
            return BacktestResult(
                ticker=ticker,
                starting_balance=starting_balance,
                risk_per_trade=risk_per_trade,
                start_date=start,
                end_date=end,
            )

        # Step 4 & 5: Convert each setup to a trade and simulate it
        trades = []
        for setup in setups:
            # Tag setup with ticker so TradeRecord knows which stock
            setup.ticker = ticker

            # Ask the strategy if this setup should be entered
            # (In M3, confirmation=None means "always enter")
            entry = self.strategy.get_entry(setup, confirmation=None)
            if entry is None:
                continue

            # Ask the strategy for stop loss and take profit levels
            exit_levels = self.strategy.get_exit(entry, setup, self.risk_config)

            # Calculate position size based on current balance and risk
            # risk_per_share = distance from entry to stop loss
            # shares = (balance × risk%) / risk_per_share
            risk_per_share = abs(entry.price - exit_levels.stop_loss)
            if risk_per_share == 0:
                # Can't size a position with zero risk — skip this trade
                continue
            risk_amount = current_balance * risk_per_trade
            shares = risk_amount / risk_per_share

            # Get 1m data for the trading day to simulate candle-by-candle
            day_str = setup.date.isoformat()
            if day_str not in data_1m.index:
                continue
            day_1m = data_1m.loc[day_str]

            # Simulate the trade (returns per-share pnl in TradeRecord)
            trade = self._simulate_trade(
                entry, exit_levels, day_1m, setup, shares, current_balance
            )
            if trade is not None:
                # Update running balance after the trade
                current_balance = trade.balance_after
                trades.append(trade)

        # Step 6: Build and return results
        result = BacktestResult(
            ticker=ticker,
            starting_balance=starting_balance,
            risk_per_trade=risk_per_trade,
            start_date=start,
            end_date=end,
            trades=trades,
        )

        return result

    def run_multi(
        self,
        tickers: list[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run a backtest across multiple tickers with a shared account balance.

        Detects setups for each ticker independently, then merges all setups
        into a single chronological timeline. Trades are processed in order
        of entry timestamp, so the shared balance compounds across stocks.

        This simulates monitoring a watchlist and taking the best setup on
        any stock each day, all from the same account.

        Args:
            tickers: List of stock ticker symbols (e.g., ["AMD", "AAPL"]).
            start: Start date in "YYYY-MM-DD" format (inclusive). Optional.
            end: End date in "YYYY-MM-DD" format (inclusive). Optional.

        Returns:
            A single BacktestResult with all trades across all tickers,
            sorted chronologically with a shared running balance.
        """
        from main import load_from_csv

        starting_balance = self.risk_config.get("starting_balance", 1000.0)
        risk_per_trade = self.risk_config.get("risk_per_trade", 0.01)
        current_balance = starting_balance

        ticker_label = ", ".join(tickers)

        print(f"\n{'=' * 60}")
        print(f"  Multi-stock backtest: {ticker_label}")
        if start and end:
            print(f"  Period: {start} to {end}")
        print(f"  Starting balance: ${starting_balance:,.2f}")
        print(f"  Risk per trade:   {risk_per_trade:.1%}")
        print(f"  R:R ratio:        {self.risk_config.get('risk_reward_ratio', 2.0)}")
        print(f"{'=' * 60}")

        # Step 1: Gather all (setup, entry, exit, ticker, day_1m) tuples
        # from every ticker so we can sort and process them chronologically
        all_trade_inputs = []

        for ticker in tickers:
            data_5m = load_from_csv(ticker, "5m")
            data_1m = load_from_csv(ticker, "1m")

            if data_5m.empty or data_1m.empty:
                print(f"  WARNING: Missing data for {ticker}, skipping.")
                continue

            # Filter to date range
            if start:
                data_5m = data_5m.loc[start:]
                data_1m = data_1m.loc[start:]
            if end:
                data_5m = data_5m.loc[:end]
                data_1m = data_1m.loc[:end]

            # Detect setups for this ticker
            setups = self.strategy.detect_setups(data_5m, data_1m)
            if not setups:
                print(f"  {ticker}: 0 setups found")
                continue

            print(f"  {ticker}: {len(setups)} setups found")

            for setup in setups:
                # Tag the setup with its ticker so TradeRecord knows
                setup.ticker = ticker

                entry = self.strategy.get_entry(setup, confirmation=None)
                if entry is None:
                    continue

                exit_levels = self.strategy.get_exit(entry, setup, self.risk_config)

                # Get 1m data for this day
                day_str = setup.date.isoformat()
                if day_str not in data_1m.index:
                    continue
                day_1m = data_1m.loc[day_str]

                all_trade_inputs.append((setup, entry, exit_levels, day_1m))

        if not all_trade_inputs:
            print("No setups found across any ticker — nothing to backtest.")
            return BacktestResult(
                ticker=ticker_label,
                starting_balance=starting_balance,
                risk_per_trade=risk_per_trade,
                start_date=start,
                end_date=end,
            )

        # Step 2: Sort all trade inputs chronologically by entry timestamp
        # This ensures the shared balance is updated in the correct order
        all_trade_inputs.sort(key=lambda x: x[1].timestamp)

        # Step 3: Process each trade in order, sizing from the shared balance
        trades = []
        for setup, entry, exit_levels, day_1m in all_trade_inputs:
            risk_per_share = abs(entry.price - exit_levels.stop_loss)
            if risk_per_share == 0:
                continue
            risk_amount = current_balance * risk_per_trade
            shares = risk_amount / risk_per_share

            trade = self._simulate_trade(
                entry, exit_levels, day_1m, setup, shares, current_balance
            )
            if trade is not None:
                current_balance = trade.balance_after
                trades.append(trade)

        return BacktestResult(
            ticker=ticker_label,
            starting_balance=starting_balance,
            risk_per_trade=risk_per_trade,
            start_date=start,
            end_date=end,
            trades=trades,
        )

    def _simulate_trade(
        self,
        entry: Entry,
        exit_levels: Exit,
        day_1m: pd.DataFrame,
        setup: "Setup" = None,
        shares: float = 1.0,
        current_balance: float = 1000.0,
    ) -> Optional[TradeRecord]:
        """
        Simulate a single trade candle-by-candle on 1-minute data.

        Starting from the candle AFTER the entry timestamp, iterates
        through each 1m candle and checks if the stop loss or take profit
        was hit. If both are hit on the same candle, uses the candle's
        open price to determine which was hit first.

        If neither SL nor TP is hit by market close, the position is
        closed at the last candle's close price ("closed_eod").

        Args:
            entry: The trade entry (price, time, direction).
            exit_levels: The stop loss and take profit levels.
            day_1m: 1-minute candle data for the trading day.

        Returns:
            A TradeRecord, or None if there are no candles after entry.
        """
        sl = exit_levels.stop_loss
        tp = exit_levels.take_profit
        is_long = entry.direction == "long"

        # Get candles AFTER the entry timestamp (we entered at entry.timestamp,
        # so we start checking from the next candle onward)
        after_entry = day_1m[day_1m.index > entry.timestamp]

        # Also filter to market hours only (up to 4:00 PM)
        after_entry = after_entry[after_entry.index.time <= MARKET_CLOSE_TIME]

        if after_entry.empty:
            # No candles after entry (entered too late in the day)
            return None

        # Walk through each candle and check for SL/TP hits
        for timestamp, candle in after_entry.iterrows():
            candle_high = float(candle["High"])
            candle_low = float(candle["Low"])
            candle_open = float(candle["Open"])
            candle_close = float(candle["Close"])

            if is_long:
                sl_hit = candle_low <= sl
                tp_hit = candle_high >= tp
            else:
                sl_hit = candle_high >= sl
                tp_hit = candle_low <= tp

            # Check if both SL and TP were hit on the same candle
            if sl_hit and tp_hit:
                # Use candle open to determine which was hit first:
                # Whichever level the open is closer to was likely hit first
                if is_long:
                    dist_to_sl = abs(candle_open - sl)
                    dist_to_tp = abs(candle_open - tp)
                else:
                    dist_to_sl = abs(candle_open - sl)
                    dist_to_tp = abs(candle_open - tp)

                if dist_to_sl <= dist_to_tp:
                    # SL was closer to open — assume loss
                    return self._build_trade_record(
                        entry, sl, timestamp, "loss", is_long, sl, tp,
                        setup, shares, current_balance
                    )
                else:
                    # TP was closer to open — assume win
                    return self._build_trade_record(
                        entry, tp, timestamp, "win", is_long, sl, tp,
                        setup, shares, current_balance
                    )

            elif sl_hit:
                return self._build_trade_record(
                    entry, sl, timestamp, "loss", is_long, sl, tp,
                    setup, shares, current_balance
                )

            elif tp_hit:
                return self._build_trade_record(
                    entry, tp, timestamp, "win", is_long, sl, tp,
                    setup, shares, current_balance
                )

        # End of day — neither SL nor TP was hit
        # Close at the last candle's close price
        last_timestamp = after_entry.index[-1]
        last_close = float(after_entry.iloc[-1]["Close"])

        return self._build_trade_record(
            entry, last_close, last_timestamp, "closed_eod", is_long, sl, tp,
            setup, shares, current_balance
        )

    def _build_trade_record(
        self,
        entry: Entry,
        exit_price: float,
        exit_time: datetime,
        result: str,
        is_long: bool,
        stop_loss: float,
        take_profit: float,
        setup: "Setup" = None,
        shares: float = 1.0,
        current_balance: float = 1000.0,
    ) -> TradeRecord:
        """
        Helper to build a TradeRecord with the correct P&L calculation.

        P&L is calculated per share first, then multiplied by position size
        to get the actual dollar P&L. The running balance is updated.

        Args:
            entry: The trade entry.
            exit_price: The price at which the trade was exited.
            exit_time: The timestamp of the exit.
            result: "win", "loss", or "closed_eod".
            is_long: True for long trades, False for short.
            stop_loss: The stop loss price level for this trade.
            take_profit: The take profit price level for this trade.
            setup: The original Setup (for opening range data).
            shares: Number of shares traded (fractional).
            current_balance: Account balance before this trade.

        Returns:
            A completed TradeRecord.
        """
        # Calculate per-share P&L
        if is_long:
            pnl = exit_price - entry.price
        else:
            pnl = entry.price - exit_price

        # Calculate actual dollar P&L based on position size
        dollar_pnl = pnl * shares
        balance_after = current_balance + dollar_pnl

        return TradeRecord(
            entry_time=entry.timestamp,
            exit_time=exit_time,
            ticker=setup.ticker if setup and hasattr(setup, 'ticker') else "",
            direction=entry.direction,
            entry_price=entry.price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pnl=pnl,
            result=result,
            or_high=setup.opening_range_high if setup else 0.0,
            or_low=setup.opening_range_low if setup else 0.0,
            shares=shares,
            dollar_pnl=dollar_pnl,
            balance_after=balance_after,
        )
