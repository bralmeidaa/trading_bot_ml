"""Shared contract for hunters: deterministic strategies that propose trades
with a built-in stop/target. The same `scan` runs live (on the latest closed
bar) and, later, in the replay simulator (bar by bar), so live and backtest
share one code path."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd


@dataclass
class Candidate:
    hunter: str
    version: str
    symbol: str
    direction: int            # 1 long / -1 short
    bar_ts: int               # ms open time of the bar that produced the signal
    ref_price: float          # close of that bar
    stop_pct: float
    target_pct: float
    rationale: str
    live: bool = True         # False → shadow only, never sent to the judge
    context: Dict[str, float] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.hunter}:{self.symbol}:{self.bar_ts}"

    @property
    def side(self) -> str:
        return "buy" if self.direction == 1 else "sell"


@dataclass
class ScanContext:
    """What a hunter may look at besides the symbol's own bars."""
    min_stop_pct: float
    max_stop_pct: float
    # (symbol, side, bucket_sec, n) -> USD per bucket, oldest first; None when no stream
    liquidations: Optional[Callable[[str, Optional[str], int, int], List[float]]] = None


class Hunter:
    name = "base"
    version = "1"
    live = True

    def scan(self, symbol: str, df: pd.DataFrame, ctx: ScanContext) -> Optional[Candidate]:
        """`df`: prepared 5m bars (backend.hunters.features.prepare), last row =
        the latest CLOSED bar. Returns at most one candidate for that bar."""
        raise NotImplementedError

    def _candidate(self, symbol: str, row: pd.Series, direction: int, stop_pct: float,
                   target_pct: float, rationale: str, **context) -> Candidate:
        return Candidate(hunter=self.name, version=self.version, symbol=symbol,
                         direction=direction, bar_ts=int(row["timestamp"]),
                         ref_price=float(row["close"]), stop_pct=stop_pct,
                         target_pct=target_pct, rationale=rationale, live=self.live,
                         context={k: round(float(v), 6) for k, v in context.items()})


def atr_stop(atr15_pct: float, ctx: ScanContext, mult: float = 1.5) -> Optional[float]:
    """Stop distance: mult × ATR(15m), floored at the cost-driven minimum and
    capped at the maximum. None if ATR is missing."""
    if atr15_pct is None or not pd.notna(atr15_pct) or atr15_pct <= 0:
        return None
    return min(max(mult * float(atr15_pct), ctx.min_stop_pct), ctx.max_stop_pct)
