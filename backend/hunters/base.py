"""Shared contract for hunters: deterministic strategies that propose trades
with a built-in stop/target. The same `scan` runs live (on the latest closed
bar) and in the replay simulator (bar by bar), so live and backtest share one
code path.

Every tunable number of a hunter is a class attribute listed in PARAMS with
the range it may take. That list is the only thing the LLM student is allowed
to change (backend/student), so a bound here is a hard limit there."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
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


# Bar features stamped on every candidate (and so on every shadow row). The
# student's filters and lessons may only reference these, plus `direction`.
STANDARD_FEATURES = ("hour", "dow", "is_weekend", "sess_asia", "sess_london", "sess_ny",
                     "vwap_dev", "atr15_pct", "ret_1h", "ret_4h", "vol_ratio")


def standard_context(row: pd.Series) -> Dict[str, float]:
    out = {}
    for k in STANDARD_FEATURES:
        v = row.get(k) if hasattr(row, "get") else None
        if v is not None and pd.notna(v):
            out[k] = round(float(v), 6)
    return out


class Hunter:
    name = "base"
    version = "1"
    live = True
    parent: Optional[str] = None          # set on student variants
    stop_atr_mult = 1.5
    target_r = 2.0
    # name -> (min, max); ints stay ints
    PARAMS: Dict[str, Tuple[float, float]] = {"stop_atr_mult": (1.0, 3.0), "target_r": (1.0, 4.0)}

    def scan(self, symbol: str, df: pd.DataFrame, ctx: ScanContext) -> Optional[Candidate]:
        """`df`: prepared 5m bars (backend.hunters.features.prepare), last row =
        the latest CLOSED bar. Returns at most one candidate for that bar."""
        raise NotImplementedError

    def prefilter(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Replay speed-up: a boolean mask over the prepared bars that is True
        wherever `scan` COULD fire (a necessary condition, never sufficient).
        None = no shortcut, scan every bar."""
        return None

    def scan_batch(self, symbol: str, df: pd.DataFrame, ctx: ScanContext) -> Optional[List[Candidate]]:
        """Replay: every candidate over the whole prepared history in one pass,
        for hunters where that is much cheaper than bar-by-bar. None = not
        implemented, the replay calls `scan` per bar."""
        return None

    def params(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in self.PARAMS}

    def with_params(self, **params) -> "Hunter":
        clone = copy.copy(self)
        for k, v in params.items():
            if k not in self.PARAMS:
                raise ValueError(f"{self.name}: unknown parameter {k!r}")
            lo, hi = self.PARAMS[k]
            if not lo <= v <= hi:
                raise ValueError(f"{self.name}: {k}={v} outside [{lo}, {hi}]")
            setattr(clone, k, type(getattr(self, k))(v))
        return clone

    def _stop(self, atr15_pct, ctx: ScanContext) -> Optional[float]:
        return atr_stop(atr15_pct, ctx, self.stop_atr_mult)

    def _candidate(self, symbol: str, row: pd.Series, direction: int, stop_pct: float,
                   target_pct: float, rationale: str, **context) -> Candidate:
        ctx = standard_context(row)
        ctx.update({k: round(float(v), 6) for k, v in context.items()})
        return Candidate(hunter=self.name, version=self.version, symbol=symbol,
                         direction=direction, bar_ts=int(row["timestamp"]),
                         ref_price=float(row["close"]), stop_pct=stop_pct,
                         target_pct=target_pct, rationale=rationale, live=self.live,
                         context=ctx)


def atr_stop(atr15_pct: float, ctx: ScanContext, mult: float = 1.5) -> Optional[float]:
    """Stop distance: mult × ATR(15m), floored at the cost-driven minimum and
    capped at the maximum. None if ATR is missing."""
    if atr15_pct is None or not pd.notna(atr15_pct) or atr15_pct <= 0:
        return None
    return min(max(mult * float(atr15_pct), ctx.min_stop_pct), ctx.max_stop_pct)
