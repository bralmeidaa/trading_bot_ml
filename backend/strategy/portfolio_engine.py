"""
Cross-sectional momentum portfolio engine (Phase 3 — paper trading).

Fully automated daily portfolio: ranks the universe by relative momentum,
holds a market-neutral basket (long top-k / short bottom-k), rebalances every
`rebalance_days`, marks-to-market between rebalances, persists trades/equity,
and trips a kill-switch on drawdown.

Decision logic is the VALIDATED core in backend.strategy.cross_sectional
(docs/STRATEGY_THESIS.md). I/O (price fetch) runs off the event loop via
asyncio.to_thread so it never stalls the API server sharing the loop.

Pure methods (compute_weights, mark_to_market, apply_rebalance, killswitch_tripped)
take data as arguments and are unit-tested without network.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.strategy.cross_sectional import (
    momentum_signal, btc_regime, target_weights, DEFAULT_COST_PER_SIDE,
)

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    universe: List[str]
    timeframe: str = "1d"
    lookback: int = 12            # bars for the momentum signal
    rebalance_days: int = 12      # rebalance cadence (in bars)
    k: int = 3                    # legs per side (long k / short k)
    mode: str = "momentum"
    btc_filter: bool = True
    max_universe: int = 15        # liquidity cap (top-N by trailing volume)
    total_capital: float = 1200.0
    cost_per_side: float = DEFAULT_COST_PER_SIDE
    max_drawdown_kill: float = 0.35   # halt if drawdown exceeds this
    mark_interval_sec: int = 3600     # mark-to-market cadence (live loop)
    history_days: int = 200           # history window to fetch (> build_panel min_bars)
    paper_trading: bool = True


@dataclass
class PortfolioState:
    weights: Dict[str, float] = field(default_factory=dict)   # current target weights
    last_prices: Dict[str, float] = field(default_factory=dict)
    equity: float = 0.0
    peak_equity: float = 0.0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    bars_since_rebalance: int = 10**9   # force rebalance on first cycle
    halted: bool = False
    rebalances: int = 0


class CrossSectionalPortfolioEngine:
    """Paper-trading portfolio engine. API-compatible read attributes exposed."""

    def __init__(self, config: PortfolioConfig, exchange=None,
                 equity_repo=None, daily_repo=None):
        self.config = config
        self.state = PortfolioState(equity=config.total_capital,
                                    peak_equity=config.total_capital)
        self.exchange = exchange
        self.equity_repo = equity_repo
        self.daily_repo = daily_repo
        self.system_start_time = datetime.now()
        self.equity_curve: List[dict] = []
        self.trade_history: List[dict] = []   # rebalance legs as trade records
        self._last_reset_date = datetime.now().date()
        self._daily_start_equity = config.total_capital

    # ── API-compatible read properties ──────────────────────────────────
    @property
    def total_pnl(self) -> float:
        return self.state.equity - self.config.total_capital

    @property
    def daily_pnl(self) -> float:
        return self.state.equity - self._daily_start_equity

    def positions(self) -> List[dict]:
        notional = self.config.total_capital
        return [
            {"symbol": s, "weight": round(w, 4),
             "side": "long" if w > 0 else "short",
             "notional": round(abs(w) * notional, 2)}
            for s, w in sorted(self.state.weights.items(), key=lambda x: -x[1])
            if abs(w) > 1e-9
        ]

    # ── Pure logic (unit-tested, no I/O) ────────────────────────────────
    def compute_weights(self, close: pd.DataFrame, volume: pd.DataFrame = None) -> pd.Series:
        """Target weights from the latest bar's signal (validated core)."""
        c = self.config
        if len(close) < c.lookback + 2:
            return pd.Series(0.0, index=close.columns)
        sig = momentum_signal(close, c.lookback, c.mode).iloc[-1]
        if c.btc_filter and not bool(btc_regime(close, c.lookback).iloc[-1]):
            return pd.Series(0.0, index=close.columns)   # regime off → flat
        liq = None
        if volume is not None:
            liq = (close * volume).rolling(30, min_periods=5).mean().iloc[-1]
        return target_weights(sig, c.k, liq, c.max_universe)

    def mark_to_market(self, prices: Dict[str, float]) -> float:
        """Update equity by the weighted return since last mark. Returns port return."""
        st, port_ret = self.state, 0.0
        if st.last_prices:
            for sym, w in st.weights.items():
                p0, p1 = st.last_prices.get(sym), prices.get(sym)
                if p0 and p1 and p0 > 0:
                    port_ret += w * (p1 / p0 - 1.0)
            st.equity *= (1.0 + port_ret)
            st.peak_equity = max(st.peak_equity, st.equity)
        st.last_prices = dict(prices)
        return port_ret

    def apply_rebalance(self, new_weights: pd.Series) -> List[dict]:
        """Switch to new target weights, charge turnover cost, return leg records."""
        st = self.state
        old = pd.Series(st.weights, dtype=float).reindex(new_weights.index).fillna(0.0)
        turnover = (new_weights - old).abs().sum()
        cost = turnover * self.config.cost_per_side
        st.equity *= (1.0 - cost)
        legs = []
        for sym, w in new_weights.items():
            if abs(w - old.get(sym, 0.0)) > 1e-9 and abs(w) > 1e-9:
                legs.append({
                    "id": f"{sym}_{int(time.time()*1000)}",
                    "symbol": sym, "weight": round(float(w), 4),
                    "side": "long" if w > 0 else "short",
                    "time": datetime.now().strftime("%H:%M"),
                    "rebalance": st.rebalances + 1,
                })
        st.weights = {s: float(w) for s, w in new_weights.items() if abs(w) > 1e-9}
        st.bars_since_rebalance = 0
        st.rebalances += 1
        return legs

    def killswitch_tripped(self) -> bool:
        st = self.state
        if st.peak_equity <= 0:
            return False
        dd = (st.peak_equity - st.equity) / st.peak_equity
        return dd >= self.config.max_drawdown_kill

    # ── Live loop (I/O via to_thread) ───────────────────────────────────
    async def start(self):
        logger.info(f"🚀 Cross-sectional portfolio engine starting "
                    f"({'PAPER' if self.config.paper_trading else 'LIVE'}, "
                    f"{len(self.config.universe)} symbols, rebalance every "
                    f"{self.config.rebalance_days} bars)")
        try:
            while not self.state.halted:
                await self._cycle()
                await asyncio.sleep(self.config.mark_interval_sec)
        except asyncio.CancelledError:
            logger.info("Portfolio engine cancelled")
        except Exception as exc:
            logger.critical(f"💥 Portfolio engine error: {exc}")

    async def _cycle(self):
        self._check_daily_reset()
        prices = await asyncio.to_thread(self._fetch_spot_prices)
        if prices:
            self.mark_to_market(prices)
            self._record_equity()

        # rebalance when due
        bars_per_day = 86400 / self.config.mark_interval_sec
        due_cycles = self.config.rebalance_days * bars_per_day
        if self.state.bars_since_rebalance >= due_cycles:
            await self._do_rebalance()
        else:
            self.state.bars_since_rebalance += 1

        if self.killswitch_tripped():
            logger.critical(f"🛑 Kill-switch: drawdown ≥ {self.config.max_drawdown_kill:.0%}. Halting.")
            self.state.halted = True

    async def _do_rebalance(self):
        close, volume = await asyncio.to_thread(self._fetch_panel)
        if close is None or close.empty:
            logger.warning("Rebalance skipped — no panel data")
            return
        weights = self.compute_weights(close, volume)
        legs = self.apply_rebalance(weights)
        self.trade_history.extend(legs)
        logger.info(f"⚖️ Rebalance #{self.state.rebalances}: {len(legs)} legs, "
                    f"equity ${self.state.equity:.2f}, positions={len(self.state.weights)}")
        self._log_snapshot(weights)

    # ── I/O helpers ─────────────────────────────────────────────────────
    def _fetch_spot_prices(self) -> Dict[str, float]:
        if self.exchange is None:
            return {}
        out = {}
        for sym in self.config.universe:
            try:
                out[sym] = float(self.exchange.fetch_ticker(sym)["last"])
            except Exception:
                pass
        return out

    def _fetch_panel(self):
        try:
            from backend.data.universe import build_panel
            # min_bars must be small enough for the live window (history_days of
            # DAILY bars) — the build_panel default (200) would drop every symbol.
            close, volume, _ = build_panel(
                symbols=self.config.universe, timeframe=self.config.timeframe,
                days=self.config.history_days, point_in_time=True,
                use_cache=False, verbose=False,
                min_bars=max(self.config.lookback * 4, 60))
            return close, volume
        except Exception as exc:
            logger.error(f"Panel fetch failed: {exc}")
            return None, None

    def _record_equity(self):
        self.equity_curve.append({
            "timestamp": int(time.time() * 1000),
            "equity": self.state.equity,
            "total_pnl": self.total_pnl,
            "daily_pnl": self.daily_pnl,
            "positions": len(self.state.weights),
        })
        if len(self.equity_curve) > 1000:
            self.equity_curve = self.equity_curve[-1000:]
        if self.equity_repo:
            try:
                self.equity_repo.save(self.state.equity, self.total_pnl,
                                      self.daily_pnl, len(self.state.weights))
            except Exception:
                pass

    def _check_daily_reset(self):
        today = datetime.now().date()
        if today != self._last_reset_date:
            if self.daily_repo:
                try:
                    self.daily_repo.save(self._last_reset_date.isoformat(),
                                         self.daily_pnl, self.state.rebalances, 0, 0)
                except Exception:
                    pass
            self._daily_start_equity = self.state.equity
            self._last_reset_date = today

    def _log_snapshot(self, weights: pd.Series):
        longs = [s for s, w in weights.items() if w > 0]
        shorts = [s for s, w in weights.items() if w < 0]
        logger.info(f"SNAPSHOT [portfolio] equity=${self.state.equity:.2f} "
                    f"dd={(self.state.peak_equity - self.state.equity)/max(self.state.peak_equity,1e-9):.1%} "
                    f"| long={longs} short={shorts}")
