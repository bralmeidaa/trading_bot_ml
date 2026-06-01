"""
Walk-forward backtesting engine.

Usage example:
    from backend.backtesting.engine import WalkForwardEngine, BacktestConfig
    from production_trading_system import OptimizedSignalGenerator

    gen = OptimizedSignalGenerator("LINK/USDT", "5m")
    engine = WalkForwardEngine()
    result = engine.run(historical_df, gen)
    print(result.sharpe_ratio, result.win_rate)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class BacktestConfig:
    commission_pct: float = 0.001   # 0.1% per side
    slippage_pct: float = 0.0005    # 0.05% per side
    initial_capital: float = 10_000.0
    n_splits: int = 5
    min_test_bars: int = 200        # minimum bars per test fold
    hold_bars: int = 3              # how many bars to hold a position


@dataclass
class TradeRecord:
    entry_bar: int
    exit_bar: int
    direction: int      # 1 long, -1 short
    entry_price: float
    exit_price: float
    pnl_pct: float      # net of costs


@dataclass
class BacktestResult:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    n_trades: int
    fold_results: List[Dict[str, Any]] = field(default_factory=list)
    is_valid: bool = False          # True if Sharpe > 0.8 in walk-forward

    def summary(self) -> str:
        return (
            f"Return={self.total_return:.2%}  Ann={self.annualized_return:.2%}  "
            f"Sharpe={self.sharpe_ratio:.2f}  Sortino={self.sortino_ratio:.2f}  "
            f"MaxDD={self.max_drawdown:.2%}  WinRate={self.win_rate:.2%}  "
            f"Trades={self.n_trades}  Valid={self.is_valid}"
        )


class WalkForwardEngine:
    """
    Runs a proper walk-forward backtest:
      - Splits historical data into n_splits train/test folds using TimeSeriesSplit
      - Trains the signal generator on each train fold
      - Simulates trades on the test fold with realistic costs
      - Aggregates all test-fold trades into portfolio metrics

    The signal generator must implement:
        initialize_from_history(df: pd.DataFrame) -> None
        generate_signals(df: pd.DataFrame) -> List[TradeSignal]
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(self, df: pd.DataFrame, signal_generator) -> BacktestResult:
        """Run walk-forward backtest and return aggregated metrics."""
        n = len(df)
        test_size = max(self.config.min_test_bars, n // (self.config.n_splits + 2))

        if n < test_size * 3:
            return self._empty_result(reason="insufficient data")

        tscv = TimeSeriesSplit(n_splits=self.config.n_splits, test_size=test_size)
        all_trades: List[TradeRecord] = []
        fold_results: List[Dict[str, Any]] = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            # Train signal generator on this fold's training data
            try:
                signal_generator.initialize_from_history(train_df)
            except Exception as e:
                fold_results.append({"fold": fold, "error": str(e)})
                continue

            # Simulate trades on test data
            fold_trades = self._simulate(test_df, signal_generator)
            all_trades.extend(fold_trades)

            fold_pnl = sum(t.pnl_pct for t in fold_trades)
            fold_results.append({
                "fold": fold,
                "n_trades": len(fold_trades),
                "pnl": fold_pnl,
                "win_rate": sum(1 for t in fold_trades if t.pnl_pct > 0) / max(len(fold_trades), 1),
            })

        return self._metrics(all_trades, fold_results)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _simulate(self, df: pd.DataFrame, signal_generator) -> List[TradeRecord]:
        """Generate signals bar-by-bar and simulate fixed-hold trades."""
        trades: List[TradeRecord] = []
        df = df.reset_index(drop=True)
        n = len(df)
        i = 0

        while i < n - self.config.hold_bars:
            # Feed all bars up to and including i as "live" context
            context = df.iloc[: i + 1].copy()
            try:
                signals = signal_generator.generate_signals(context)
            except Exception:
                i += 1
                continue

            if not signals:
                i += 1
                continue

            sig = signals[-1]
            direction = sig.direction

            # Entry at next bar's open (conservative; use close if open not available)
            entry_bar = i + 1
            if entry_bar >= n:
                break

            raw_entry = df.iloc[entry_bar].get("open", df.iloc[entry_bar]["close"])
            entry_price = self._apply_costs(raw_entry, "buy" if direction == 1 else "sell")

            # Exit after hold_bars
            exit_bar = min(entry_bar + self.config.hold_bars, n - 1)
            raw_exit = df.iloc[exit_bar].get("open", df.iloc[exit_bar]["close"])
            exit_price = self._apply_costs(raw_exit, "sell" if direction == 1 else "buy")

            pnl_pct = (exit_price - entry_price) / entry_price * direction
            trades.append(TradeRecord(entry_bar, exit_bar, direction, entry_price, exit_price, pnl_pct))

            i = exit_bar  # skip to after the hold period

        return trades

    def _apply_costs(self, price: float, side: str) -> float:
        cost = self.config.commission_pct + self.config.slippage_pct
        return price * (1 + cost) if side == "buy" else price * (1 - cost)

    def _metrics(self, trades: List[TradeRecord], fold_results: List[Dict]) -> BacktestResult:
        if not trades:
            return self._empty_result(fold_results=fold_results)

        returns = np.array([t.pnl_pct for t in trades])
        wins = returns[returns > 0]
        losses = returns[returns <= 0]

        # Cumulative performance
        equity = np.cumprod(1 + returns)
        total_return = float(equity[-1] - 1)

        # Annualise assuming ~252 trading days; use trade count as proxy for frequency
        n = len(returns)
        ann_factor = 252 / max(n, 1) if n > 0 else 1
        annualized = float((1 + total_return) ** ann_factor - 1)

        # Risk metrics
        vol = float(returns.std() * np.sqrt(252))
        sharpe = annualized / vol if vol > 0 else 0.0

        downside = returns[returns < 0]
        down_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else vol
        sortino = annualized / down_vol if down_vol > 0 else 0.0

        # Max drawdown
        roll_max = np.maximum.accumulate(equity)
        drawdown = (equity - roll_max) / roll_max
        max_dd = float(abs(drawdown.min()))

        calmar = annualized / max_dd if max_dd > 0 else 0.0

        win_rate = float(len(wins) / n) if n > 0 else 0.0
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 1.0
        profit_factor = (avg_win * len(wins)) / (avg_loss * max(len(losses), 1))

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            n_trades=n,
            fold_results=fold_results,
            is_valid=sharpe > 0.8 and max_dd < 0.25,
        )

    @staticmethod
    def _empty_result(reason: str = "", fold_results: List[Dict] | None = None) -> BacktestResult:
        return BacktestResult(
            total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0,
            sortino_ratio=0.0, max_drawdown=0.0, calmar_ratio=0.0,
            win_rate=0.0, profit_factor=0.0, n_trades=0,
            fold_results=fold_results or [], is_valid=False,
        )
