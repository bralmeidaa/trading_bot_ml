from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
from sqlalchemy.orm import Session

from ..models.market import Candle
from ..utils.indicators import compute_indicators
from ..strategy.base import BaseStrategy


@dataclass
class BacktestResult:
    trades: List[Dict]
    equity_curve: List[Dict]
    metrics: Dict


class Backtester:
    """
    Minimal backtesting scaffold. Loads candles from DB, computes indicators,
    and runs a trivial strategy placeholder to validate the pipeline.
    """

    def __init__(self, db: Session, symbol: str, timeframe: str = "1m"):
        self.db = db
        self.symbol = symbol
        self.timeframe = timeframe

    def load_candles(self) -> pd.DataFrame:
        q = (
            self.db.query(Candle)
            .filter(Candle.symbol == self.symbol, Candle.timeframe == self.timeframe)
            .order_by(Candle.ts.asc())
        )
        rows = [
            {
                "ts": r.ts,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in q.all()
        ]
        return pd.DataFrame(rows)

    def run(self) -> BacktestResult:
        df = self.load_candles()
        return self.run_on_df(df)

    @staticmethod
    def run_on_df(df: pd.DataFrame) -> BacktestResult:
        if df.empty:
            return BacktestResult(trades=[], equity_curve=[], metrics={"message": "no data"})

        df = compute_indicators(df)

        # Strategy: cross close > ema20 => long; exit when close < ema20
        position = 0
        entry_price = 0.0
        equity = 1000.0
        qty = 0.0
        trades: List[Dict] = []
        curve: List[Dict] = []

        for _, row in df.iterrows():
            price = float(row["close"])  # type: ignore
            ema20_val = row.get("ema20")
            ema20 = float(ema20_val) if pd.notna(ema20_val) else float("nan")
            ts = int(row["ts"])  # type: ignore

            if position == 0 and pd.notna(ema20) and price > ema20:
                qty = equity / price
                entry_price = price
                position = 1
                trades.append({"ts": ts, "side": "BUY", "price": price, "qty": qty})
            elif position == 1 and pd.notna(ema20) and price < ema20:
                equity = qty * price
                trades.append({"ts": ts, "side": "SELL", "price": price, "qty": qty})
                position = 0
                qty = 0
                entry_price = 0

            mark_equity = equity if position == 0 else qty * price
            curve.append({"ts": ts, "equity": mark_equity})

        if position == 1 and not df.empty:
            price = float(df.iloc[-1]["close"])  # type: ignore
            equity = qty * price
            trades.append({"ts": int(df.iloc[-1]["ts"]), "side": "SELL", "price": price, "qty": qty})

        start_equity = 1000.0
        pnl = equity - start_equity
        returns = (equity / start_equity) - 1.0
        win_trades = 0
        loss_trades = 0
        for i in range(0, len(trades), 2):
            if i + 1 < len(trades):
                buy = trades[i]
                sell = trades[i + 1]
                if sell["price"] > buy["price"]:
                    win_trades += 1
                else:
                    loss_trades += 1
        total_trades = (len(trades)) // 2
        win_rate = (win_trades / total_trades) * 100 if total_trades else 0

        metrics = {
            "start_equity": start_equity,
            "end_equity": equity,
            "pnl": pnl,
            "return_pct": returns * 100,
            "trades": total_trades,
            "win_rate_pct": win_rate,
        }

        return BacktestResult(trades=trades, equity_curve=curve, metrics=metrics)

    @staticmethod
    def run_strategy_on_df(df: pd.DataFrame, strategy: BaseStrategy) -> BacktestResult:
        if df.empty:
            return BacktestResult(trades=[], equity_curve=[], metrics={"message": "no data"})
        # Strategy implementation is responsible for indicators/feature prep as needed
        return strategy.run(df)
