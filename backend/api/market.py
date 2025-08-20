from fastapi import APIRouter, HTTPException, Query, Form
from typing import List, Literal
from ..data.binance_client import BinancePublicClient
from ..data.db import SessionLocal
from ..models.market import Candle
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from datetime import datetime, timezone
import logging

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/klines")
async def get_klines(
    symbol: str = Query(..., description="Trading pair, e.g., BTC/USDT"),
    timeframe: Literal[
        "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"
    ] = "1m",
    limit: int = Query(100, ge=1, le=1000),
) -> List[list]:
    """Return OHLCV for the given symbol and timeframe using Binance (spot)."""
    try:
        client = BinancePublicClient()
        data = client.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/backfill")
async def backfill_market_data(
    symbol: str = Form(default=None),
    timeframe: Literal[
        "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"
    ] = Form(default="1m"),
    limit: int = Form(default=500),
):
    """
    Backfill candles for a symbol/timeframe using ccxt. Accepts form-data for the HTML UI.
    Uses SQLite upsert to ignore duplicates.
    """
    try:
        if not symbol:
            raise ValueError("symbol is required")
        client = BinancePublicClient()
        data = client.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        inserted = 0
        skipped = 0
        with SessionLocal() as db:
            for ts, o, h, l, c, v in data:
                try:
                    stmt = sqlite_insert(Candle).values(
                        symbol=symbol,
                        timeframe=timeframe,
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
                    inserted += 1
                except Exception as ex:
                    skipped += 1
                    logging.warning(
                        f"[backfill] conflict or error symbol={symbol} tf={timeframe} ts={ts}: {ex}"
                    )
            db.commit()
        return {"status": "ok", "requested": limit, "inserted_attempted": inserted, "skipped": skipped}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/candles")
async def get_candles(
    symbol: str = Query(..., description="Trading pair, e.g., BTC/USDT"),
    timeframe: Literal[
        "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"
    ] = "1m",
    limit: int = Query(100, ge=1, le=2000),
):
    """Return stored candles from DB ordered by time, including ISO timestamp field."""
    try:
        with SessionLocal() as db:
            q = (
                db.query(Candle)
                .filter(Candle.symbol == symbol, Candle.timeframe == timeframe)
                .order_by(Candle.ts.desc())
                .limit(limit)
            )
            rows = q.all()
            result = []
            for r in reversed(rows):
                iso = datetime.fromtimestamp(r.ts / 1000, tz=timezone.utc).isoformat()
                result.append(
                    {
                        "id": r.id,
                        "symbol": r.symbol,
                        "timeframe": r.timeframe,
                        "ts": r.ts,
                        "ts_iso": iso,
                        "open": r.open,
                        "high": r.high,
                        "low": r.low,
                        "close": r.close,
                        "volume": r.volume,
                    }
                )
            return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
