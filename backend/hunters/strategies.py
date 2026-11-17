"""
The hunters. Each looks only at the latest closed 5m bar (and the history
before it) and proposes at most one trade with its own stop/target.

None of these setups was tested in earlier research except the control
(OrbControl), which is a known NO-GO and must keep losing in the shadow
ledger — if it starts "winning", the measurement is wrong, not the strategy.

Stops are ATR(15m)-based and never tighter than the cost-driven floor
(ScanContext.min_stop_pct): the 5m ORB research died because a 1×ATR(5m)
stop made round-trip costs ~0.6R per trade.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from backend.hunters.base import Candidate, Hunter, ScanContext
from backend.strategy.intraday_setups import SetupConfig, generate_setups


def _ok(*values) -> bool:
    return all(v is not None and pd.notna(v) for v in values)


class VwapReversion(Hunter):
    """A flush far from the day's VWAP that turns back on the same bar:
    - stretched: beyond the 2σ band AND ≥ 4×ATR(15m) from VWAP (in a steady
      trend price sits outside a 20-bar band for hours, so the band alone
      fired ~780×/day across 15 coins);
    - fresh extreme: the bar made the lowest low (highest high) of the last 2h;
    - turn: closes up (down) vs its open and beyond the prior bar's high
      (low) — an engulfing reversal — in the upper (lower) half of its range.
    Target: the VWAP itself; skipped unless that's at least 1R away."""
    name, version = "vwap_reversion", "1"
    stretch_atr, lookback_bars, min_target_r = 4.0, 24, 1.0
    PARAMS = {"stretch_atr": (2.5, 6.0), "lookback_bars": (12, 48),
              "min_target_r": (1.0, 2.0), "stop_atr_mult": (1.0, 3.0)}

    def prefilter(self, df):
        return ((df["low"] < df["vwap_lower"]) | (df["high"] > df["vwap_upper"])).to_numpy()

    def scan(self, symbol, df, ctx) -> Optional[Candidate]:
        if len(df) < self.lookback_bars + 2:
            return None
        row, prev = df.iloc[-1], df.iloc[-2]
        vwap, lo, hi, atr = row["vwap"], row["vwap_lower"], row["vwap_upper"], row["atr15_pct"]
        stop = self._stop(atr, ctx)
        if not _ok(vwap, lo, hi, stop):
            return None
        close, rng = row["close"], row["high"] - row["low"]
        recent = df.iloc[-self.lookback_bars:]
        if rng <= 0:
            return None
        if (row["low"] < lo and (vwap - row["low"]) / vwap >= self.stretch_atr * atr
                and row["low"] <= recent["low"].min()
                and close > row["open"] and close > prev["high"]
                and close >= row["low"] + rng / 2):
            direction = 1
        elif (row["high"] > hi and (row["high"] - vwap) / vwap >= self.stretch_atr * atr
                and row["high"] >= recent["high"].max()
                and close < row["open"] and close < prev["low"]
                and close <= row["high"] - rng / 2):
            direction = -1
        else:
            return None
        target = abs(vwap - close) / close
        if target < self.min_target_r * stop:
            return None
        return self._candidate(symbol, row, direction, stop, target,
                               f"{'below' if direction == 1 else 'above'} the 2σ VWAP band and turning "
                               f"back; target = VWAP ({target:.2%} away), stop {stop:.2%}",
                               vwap_dev=row["vwap_dev"], atr15_pct=row["atr15_pct"])


class SessionBreakout(Hunter):
    """First close beyond the 30-min opening range of the London or New York
    session (within its first 3h), on the VWAP side of the break. 2R target."""
    version = "1"

    def __init__(self, session: str):
        self.session = session
        self.name = f"session_breakout_{session}"

    def prefilter(self, df):
        s = self.session
        return (df[f"in_window_{s}"].astype(bool)
                & ((df["close"] > df[f"or_high_{s}"]) | (df["close"] < df[f"or_low_{s}"]))).to_numpy()

    def scan(self, symbol, df, ctx) -> Optional[Candidate]:
        if len(df) < 2:
            return None
        row, prev = df.iloc[-1], df.iloc[-2]
        s = self.session
        hi, lo = row[f"or_high_{s}"], row[f"or_low_{s}"]
        if not row[f"in_window_{s}"] or not _ok(hi, lo):
            return None
        stop = self._stop(row["atr15_pct"], ctx)
        if stop is None:
            return None
        close, vwap = row["close"], row["vwap"]
        # only the FIRST close beyond the range this session counts; re-crosses are chop
        earlier = df.iloc[:-1]
        earlier = earlier[(earlier["_date"] == row["_date"]) & earlier[f"in_window_{s}"]]
        if close > hi and prev["close"] <= hi and close > vwap and not (earlier["close"] > hi).any():
            direction = 1
        elif close < lo and prev["close"] >= lo and close < vwap and not (earlier["close"] < lo).any():
            direction = -1
        else:
            return None
        return self._candidate(symbol, row, direction, stop, self.target_r * stop,
                               f"first close {'above' if direction == 1 else 'below'} the {s} "
                               f"30-min opening range ({lo:.6g}–{hi:.6g}), VWAP-aligned",
                               or_width_pct=(hi - lo) / close, atr15_pct=row["atr15_pct"])


class PriorDayFalseBreak(Hunter):
    """Price pushed past the prior day's high (low), then a bar closed back
    inside: trapped breakout traders → fade it. 2R target."""
    name, version = "prior_day_false_break", "1"

    def prefilter(self, df):
        pc = df["close"].shift(1)
        return (((pc > df["pdh"]) & (df["close"] < df["pdh"]))
                | ((pc < df["pdl"]) & (df["close"] > df["pdl"]))).to_numpy()

    def scan(self, symbol, df, ctx) -> Optional[Candidate]:
        if len(df) < 2:
            return None
        row, prev = df.iloc[-1], df.iloc[-2]
        pdh, pdl = row["pdh"], row["pdl"]
        stop = self._stop(row["atr15_pct"], ctx)
        if not _ok(pdh, pdl, stop):
            return None
        close = row["close"]
        if prev["close"] > pdh and close < pdh:
            direction = -1
        elif prev["close"] < pdl and close > pdl:
            direction = 1
        else:
            return None
        level = pdh if direction == -1 else pdl
        return self._candidate(symbol, row, direction, stop, self.target_r * stop,
                               f"closed back {'below the prior-day high' if direction == -1 else 'above the prior-day low'} "
                               f"({level:.6g}) after breaking it",
                               level=level, atr15_pct=row["atr15_pct"])


class LiquidationReversal(Hunter):
    """A liquidation cascade (≥ 3× the median of the previous 2h, in 5-min
    buckets) plus a ≥ 1.5×ATR move in the last 15 min → fade the forced
    flow. Live only: needs the WebSocket liquidation feed."""
    name, version = "liquidation_reversal", "1"
    burst_mult, min_usd, move_atr_mult = 3.0, 100_000.0, 1.5
    PARAMS = {"burst_mult": (2.0, 6.0), "move_atr_mult": (1.0, 3.0),
              "stop_atr_mult": (1.0, 3.0), "target_r": (1.0, 4.0)}

    def prefilter(self, df):
        move = (df["close"] / df["close"].shift(3) - 1).abs()
        return (move >= self.move_atr_mult * df["atr15_pct"]).to_numpy()

    def scan(self, symbol, df, ctx) -> Optional[Candidate]:
        if ctx.liquidations is None or len(df) < 4:
            return None
        row = df.iloc[-1]
        stop = self._stop(row["atr15_pct"], ctx)
        if stop is None:
            return None
        move = (row["close"] - df.iloc[-4]["close"]) / df.iloc[-4]["close"]
        if abs(move) < self.move_atr_mult * row["atr15_pct"]:
            return None
        # a drop liquidates longs (forced SELL orders); a spike liquidates shorts (BUY)
        side = "SELL" if move < 0 else "BUY"
        buckets = ctx.liquidations(symbol, side, 300, 25)
        if len(buckets) < 13:
            return None
        current, baseline = buckets[-1], float(np.median(buckets[:-1]))
        if current < self.min_usd or current < self.burst_mult * max(baseline, 1.0):
            return None
        direction = 1 if move < 0 else -1
        return self._candidate(symbol, row, direction, stop, self.target_r * stop,
                               f"liquidation cascade: ${current / 1e6:.2f}M of {'longs' if move < 0 else 'shorts'} "
                               f"liquidated in 5 min (median ${baseline / 1e6:.2f}M) after a {move:+.2%} move — "
                               f"fading the forced flow",
                               liq_usd=current, liq_median=baseline, move_15m=move)


class OrbControl(Hunter):
    """CONTROL — the 5m ORB+VWAP+trend setup that research already showed
    loses after costs (1×ATR(5m) stop). Shadow only, never traded."""
    name, version, live = "orb_control", "1", False
    PARAMS = {}                           # a control is never tuned

    def scan_batch(self, symbol, df, ctx) -> List[Candidate]:
        """Replay: one generate_setups pass over the whole history instead of
        one per bar (its indicators are trailing, so the result matches)."""
        bars = df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        out = []
        for st in generate_setups(bars, symbol, SetupConfig()):
            out.append(self._candidate(symbol, df.iloc[st.bar_index], st.direction,
                                       abs(st.entry - st.stop) / st.entry,
                                       abs(st.target - st.entry) / st.entry,
                                       "control: 5m opening-range breakout (known NO-GO)"))
        return out

    def scan(self, symbol, df, ctx) -> Optional[Candidate]:
        if len(df) < 60:
            return None
        bars = df[["timestamp", "open", "high", "low", "close", "volume"]]
        setups = generate_setups(bars, symbol, SetupConfig())
        last = len(bars) - 1
        hit = next((s for s in setups if s.bar_index == last), None)
        if hit is None:
            return None
        stop_pct = abs(hit.entry - hit.stop) / hit.entry
        target_pct = abs(hit.target - hit.entry) / hit.entry
        return self._candidate(symbol, df.iloc[-1], hit.direction, stop_pct, target_pct,
                               "control: 5m opening-range breakout (known NO-GO)")


def default_hunters() -> List[Hunter]:
    return [VwapReversion(), SessionBreakout("london"), SessionBreakout("ny"),
            PriorDayFalseBreak(), LiquidationReversal(), OrbControl()]


# Base hunters by name — what student variants are built from.
BASE_HUNTERS = {
    "vwap_reversion": VwapReversion,
    "session_breakout_london": lambda: SessionBreakout("london"),
    "session_breakout_ny": lambda: SessionBreakout("ny"),
    "prior_day_false_break": PriorDayFalseBreak,
    "liquidation_reversal": LiquidationReversal,
}
