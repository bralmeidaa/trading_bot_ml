from __future__ import annotations
from typing import List, Dict
import pandas as pd
import numpy as np

from .base import BaseStrategy
from ..backtesting.engine import BacktestResult
from ..utils.indicators import compute_indicators
from ..core.risk import position_size
from ..utils.backtest_metrics import calc_max_drawdown, calc_sharpe


class RuleEmaStrategy(BaseStrategy):
    """
    Simple EMA crossover strategy for short-term trading.
    - Entry: EMA12 crosses above EMA26 -> go long
    - Exit: EMA12 crosses below EMA26 -> exit
    Optional: Uses risk-based sizing with fixed equity base of 1000 for backtest comparability.
    """

    def __init__(self, risk_per_trade_pct: float = 1.0, fee_bps: float = 10.0, slippage_bps: float = 1.0):
        self.risk_per_trade_pct = risk_per_trade_pct
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps

    def run(self, df: pd.DataFrame) -> BacktestResult:
        if df.empty:
            return BacktestResult(trades=[], equity_curve=[], metrics={"message": "no data"})

        df = compute_indicators(df)
        # Add EMA12/EMA26
        df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
        df.dropna(inplace=True)

        position = 0
        entry_price = 0.0
        start_equity = 1000.0
        cash = start_equity
        qty = 0.0
        trades: List[Dict] = []
        curve: List[Dict] = []

        prev_diff = None
        for _, row in df.iterrows():
            price = float(row["close"])  # type: ignore
            ts = int(row["ts"])  # type: ignore
            ema12 = float(row["ema12"])  # type: ignore
            ema26 = float(row["ema26"])  # type: ignore

            diff = ema12 - ema26
            cross_up = prev_diff is not None and prev_diff <= 0 and diff > 0
            cross_down = prev_diff is not None and prev_diff >= 0 and diff < 0

            if position == 0 and cross_up:
                # allocate a fraction of current cash
                alloc_cash = cash * (self.risk_per_trade_pct / 100.0)
                buy_price = price * (1.0 + (self.fee_bps + self.slippage_bps) / 10000.0)
                if buy_price > 0 and alloc_cash > 0:
                    qty = alloc_cash / buy_price
                    cash -= qty * buy_price
                    entry_price = buy_price
                    position = 1
                    trades.append({"ts": ts, "side": "BUY", "price": buy_price, "qty": qty})
            elif position == 1 and cross_down:
                sell_price = price * (1.0 - (self.fee_bps + self.slippage_bps) / 10000.0)
                cash += qty * sell_price
                trades.append({"ts": ts, "side": "SELL", "price": sell_price, "qty": qty})
                position = 0
                qty = 0.0
                entry_price = 0.0

            mark_equity = cash + (qty * price if position == 1 else 0.0)
            curve.append({"ts": ts, "equity": mark_equity})
            prev_diff = diff

        if position == 1 and not df.empty:
            price = float(df.iloc[-1]["close"])  # type: ignore
            sell_price = price * (1.0 - (self.fee_bps + self.slippage_bps) / 10000.0)
            cash += qty * sell_price
            trades.append({"ts": int(df.iloc[-1]["ts"]), "side": "SELL", "price": sell_price, "qty": qty})

        end_equity = cash
        pnl = end_equity - start_equity
        returns = (end_equity / start_equity) - 1.0

        # Compute win rate from paired trades
        win_trades = 0
        total_trades = 0
        for i in range(0, len(trades), 2):
            if i + 1 < len(trades):
                buy = trades[i]
                sell = trades[i + 1]
                total_trades += 1
                if sell["price"] > buy["price"]:
                    win_trades += 1
        win_rate = (win_trades / total_trades) * 100 if total_trades else 0

        metrics = {
            "start_equity": start_equity,
            "end_equity": end_equity,
            "pnl": pnl,
            "return_pct": returns * 100,
            "trades": total_trades,
            "win_rate_pct": win_rate,
            "max_drawdown_pct": calc_max_drawdown(curve),
            "sharpe": calc_sharpe(curve),
        }

        return BacktestResult(trades=trades, equity_curve=curve, metrics=metrics)
