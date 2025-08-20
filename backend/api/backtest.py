from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Literal
import logging
import pandas as pd

from ..data.db import SessionLocal
from ..backtesting.engine import Backtester
from ..data.binance_client import BinancePublicClient
from ..strategy.rule_ema import RuleEmaStrategy
from ..strategy.ml_prob import MLProbStrategy

router = APIRouter(prefix="/backtest", tags=["backtest"])


class BacktestResponse(BaseModel):
    metrics: dict
    trades: list
    equity_curve: list


@router.get("/")
async def run_backtest(
    symbol: str = Query(..., description="Symbol like BTC/USDT"),
    timeframe: Literal[
        "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"
    ] = "1m",
):
    try:
        with SessionLocal() as db:
            logging.info(f"[backtest] DB mode start symbol={symbol} tf={timeframe}")
            bt = Backtester(db=db, symbol=symbol, timeframe=timeframe)
            res = bt.run()
            logging.info(
                f"[backtest] DB mode end symbol={symbol} tf={timeframe} trades={len(res.trades)} curve={len(res.equity_curve)}"
            )
            return BacktestResponse(metrics=res.metrics, trades=res.trades, equity_curve=res.equity_curve)
    except Exception as e:
        logging.exception("[backtest] DB mode error")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/inmemory")
async def run_backtest_inmemory(
    symbol: str = Query(..., description="Symbol like BTC/USDT"),
    timeframe: Literal[
        "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"
    ] = "1m",
    limit: int = Query(500, ge=50, le=2000),
):
    """Run a backtest entirely in-memory from live klines (does not touch DB)."""
    try:
        logging.info(f"[backtest] INMEMORY start symbol={symbol} tf={timeframe} limit={limit}")
        client = BinancePublicClient()
        data = client.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        if not data:
            return BacktestResponse(metrics={"message": "no data"}, trades=[], equity_curve=[])
        # data entries: [ms, open, high, low, close, volume]
        df = pd.DataFrame(
            data,
            columns=["ts", "open", "high", "low", "close", "volume"],
        )
        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)
        res = Backtester.run_on_df(df)
        logging.info(
            f"[backtest] INMEMORY end symbol={symbol} tf={timeframe} trades={len(res.trades)} curve={len(res.equity_curve)}"
        )
        return BacktestResponse(metrics=res.metrics, trades=res.trades, equity_curve=res.equity_curve)
    except Exception as e:
        logging.exception("[backtest] INMEMORY error")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/inmemory/strategy")
async def run_backtest_inmemory_strategy(
    symbol: str = Query(..., description="Symbol like BTC/USDT"),
    timeframe: Literal[
        "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"
    ] = "1m",
    limit: int = Query(800, ge=100, le=3000),
    name: Literal["rule_ema", "ml_prob"] = "ml_prob",
    # Common params
    risk_per_trade_pct: float = 1.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 1.0,
    # ML params
    threshold: float = 0.6,
    exit_threshold: float = 0.5,
    train_ratio: float = 0.7,
    min_atr_pct: float = 0.001,
    max_atr_pct: float = 0.05,
    max_hold_bars: int = 60,
    # Advanced ML params
    label_horizon: int = 5,
    label_cost_bp: float = 15.0,
    auto_threshold: bool = True,
    use_regime_percentiles: bool = True,
    regime_low_pctile: float = 20.0,
    regime_high_pctile: float = 90.0,
    sl_atr_mult: float = 1.2,
    tp_atr_mult: float = 2.0,
    min_val_trades: int = 3,
    calibrate: str | None = None,  # 'platt' | 'isotonic' | None
    calibrate_cv: int = 3,
    min_test_signals: int = 5,
    test_threshold_floor: float = 0.03,
):
    """Run a pluggable strategy in-memory using live klines (no DB access)."""
    try:
        logging.info(
            f"[backtest] STRAT start name={name} symbol={symbol} tf={timeframe} limit={limit}"
        )
        client = BinancePublicClient()
        data = client.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        if not data:
            return BacktestResponse(metrics={"message": "no data"}, trades=[], equity_curve=[])

        df = pd.DataFrame(
            data,
            columns=["ts", "open", "high", "low", "close", "volume"],
        )
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)

        # Select strategy
        if name == "rule_ema":
            strat = RuleEmaStrategy(
                risk_per_trade_pct=risk_per_trade_pct,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
            )
        elif name == "ml_prob":
            strat = MLProbStrategy(
                threshold=threshold,
                exit_threshold=exit_threshold,
                train_ratio=train_ratio,
                risk_per_trade_pct=risk_per_trade_pct,
                min_atr_pct=min_atr_pct,
                max_atr_pct=max_atr_pct,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                max_hold_bars=max_hold_bars,
                label_horizon=label_horizon,
                label_cost_bp=label_cost_bp,
                auto_threshold=auto_threshold,
                use_regime_percentiles=use_regime_percentiles,
                regime_low_pctile=regime_low_pctile,
                regime_high_pctile=regime_high_pctile,
                sl_atr_mult=sl_atr_mult,
                tp_atr_mult=tp_atr_mult,
                min_val_trades=min_val_trades,
                calibrate=calibrate,
                calibrate_cv=calibrate_cv,
                min_test_signals=min_test_signals,
                test_threshold_floor=test_threshold_floor,
            )
        else:
            raise HTTPException(status_code=400, detail=f"unknown strategy: {name}")

        res = Backtester.run_strategy_on_df(df, strat)
        logging.info(
            f"[backtest] STRAT end name={name} symbol={symbol} tf={timeframe} trades={len(res.trades)} curve={len(res.equity_curve)}"
        )
        return BacktestResponse(metrics=res.metrics, trades=res.trades, equity_curve=res.equity_curve)
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("[backtest] STRAT error")
        raise HTTPException(status_code=400, detail=str(e))
