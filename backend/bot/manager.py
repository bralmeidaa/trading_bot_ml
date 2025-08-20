import asyncio
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime

from ..data.binance_client import BinancePublicClient
from ..data.db import SessionLocal
from ..models.market import Candle
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
import pandas as pd
from ..utils.indicators import compute_indicators
from ..core.risk import position_size, compute_sl_tp, RiskParams
import logging


@dataclass
class BotConfig:
    symbol: str
    timeframe: str = "1m"
    poll_seconds: int = 5


@dataclass
class BotState:
    id: str
    config: BotConfig
    started_at: datetime
    task: asyncio.Task
    # Paper trading state
    equity: float = 1000.0
    position: int = 0  # 0 flat, 1 long
    qty: float = 0.0
    entry_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    last_ts: Optional[int] = None
    equity_curve: List[Dict] = None  # list of {ts, equity}


class BotManager:
    def __init__(self):
        self._bots: Dict[str, BotState] = {}
        self._client = BinancePublicClient()

    def list(self):
        return [
            {
                "id": b.id,
                "symbol": b.config.symbol,
                "timeframe": b.config.timeframe,
                "poll_seconds": b.config.poll_seconds,
                "started_at": b.started_at.isoformat(),
                "running": not b.task.done(),
                "equity": getattr(b, "equity", None),
                "position": getattr(b, "position", 0),
            }
            for b in self._bots.values()
        ]

    def is_running(self, bot_id: str) -> bool:
        return bot_id in self._bots and not self._bots[bot_id].task.done()

    async def _worker(self, bot_id: str, cfg: BotConfig):
        risk = RiskParams()
        # initialize state container
        state = self._bots[bot_id]
        state.equity_curve = []
        while True:
            try:
                # Fetch recent candles and persist newest
                data = await asyncio.to_thread(
                    self._client.fetch_ohlcv, cfg.symbol, cfg.timeframe, 1
                )
                if data:
                    ts, o, h, l, c, v = data[-1]
                    with SessionLocal() as db:
                        # Upsert (ignore duplicates) for SQLite
                        try:
                            stmt = sqlite_insert(Candle).values(
                                symbol=cfg.symbol,
                                timeframe=cfg.timeframe,
                                ts=ts,
                                open=o,
                                high=h,
                                low=l,
                                close=c,
                                volume=v,
                            )
                            stmt = stmt.on_conflict_do_nothing(
                                index_elements=[Candle.symbol, Candle.timeframe, Candle.ts]
                            )
                            db.execute(stmt)
                            db.commit()
                        except Exception as ex:
                            logging.warning(
                                f"[bot] insert conflict/error id={bot_id} symbol={cfg.symbol} tf={cfg.timeframe} ts={ts}: {ex}"
                            )

                        # Load last 50 candles for decision making
                        q = (
                            db.query(Candle)
                            .filter(Candle.symbol == cfg.symbol, Candle.timeframe == cfg.timeframe)
                            .order_by(Candle.ts.desc())
                            .limit(50)
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
                        rows.reverse()
                        df = pd.DataFrame(rows)
                        if not df.empty:
                            df = compute_indicators(df)
                            last = df.iloc[-1]
                            price = float(last["close"])  # current price
                            ema20 = last.get("ema20")

                            # Check exits first (SL/TP)
                            if state.position == 1:
                                if price <= state.sl or price >= state.tp:
                                    # close long
                                    state.equity = state.qty * price
                                    state.position = 0
                                    state.qty = 0.0
                                    state.entry_price = 0.0
                                    state.sl = 0.0
                                    state.tp = 0.0

                            # Simple entry rule: price > ema20 => long if flat
                            if state.position == 0 and pd.notna(ema20) and price > float(ema20):
                                qty = position_size(state.equity, price, risk.risk_per_trade_pct)
                                if qty > 0:
                                    state.qty = qty
                                    state.entry_price = price
                                    state.position = 1
                                    sl, tp = compute_sl_tp(price, risk.stop_loss_pct, risk.take_profit_pct)
                                    state.sl, state.tp = sl, tp

                            # mark-to-market equity
                            mark_equity = state.equity if state.position == 0 else state.qty * price
                            state.last_ts = int(last["ts"])  # type: ignore
                            state.equity_curve.append({"ts": state.last_ts, "equity": mark_equity})
                await asyncio.sleep(cfg.poll_seconds)
            except asyncio.CancelledError:
                break
            except Exception:
                # Backoff on unexpected errors
                await asyncio.sleep(max(1, cfg.poll_seconds))

    def start(self, bot_id: str, symbol: str, timeframe: str = "1m", poll_seconds: int = 5) -> Dict:
        if self.is_running(bot_id):
            return {"status": "already_running", "id": bot_id}
        cfg = BotConfig(symbol=symbol, timeframe=timeframe, poll_seconds=poll_seconds)
        task = asyncio.create_task(self._worker(bot_id, cfg))
        state = BotState(id=bot_id, config=cfg, started_at=datetime.utcnow(), task=task)
        self._bots[bot_id] = state
        return {"status": "started", "id": bot_id}

    def stop(self, bot_id: str) -> Dict:
        if bot_id not in self._bots:
            return {"status": "not_found", "id": bot_id}
        task = self._bots[bot_id].task
        if not task.done():
            task.cancel()
        return {"status": "stopping", "id": bot_id}


# Singleton manager for app scope
manager = BotManager()
