"""
Intraday setups — the "well-marked strategy" (estratégia bem marcada).

Encodes the evidence-backed Opening Range Breakout + VWAP + multi-timeframe
trend + session filter as EXPLICIT, testable setup events (not per-bar
prediction). Each setup carries entry, stop and target with asymmetric R:R.

A setup fires on the FIRST bar that breaks the opening range in the trend
direction, confirmed by VWAP, session and volume. The AI meta-filter
(research_intraday.py) then decides which setups to actually take.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from backend.data.intraday_features import build_intraday_features, CONTEXT_FEATURES


@dataclass
class SetupConfig:
    or_minutes: int = 30
    bar_minutes: int = 5
    rr: float = 2.0                  # reward:risk (target = rr × risk)
    stop_atr_mult: float = 1.0       # stop distance = mult × ATR
    require_trend: bool = True       # entry must align with multi-TF trend
    require_vwap: bool = True        # long above VWAP / short below
    min_vol_ratio: float = 1.0       # volume confirmation
    allow_sessions: tuple = ("sess_london", "sess_ny")  # trade only these
    block_weekend: bool = True


@dataclass
class Setup:
    ts: int
    symbol: str
    direction: int          # 1 long, -1 short
    entry: float
    stop: float
    target: float
    rr: float
    bar_index: int
    context: Dict[str, float] = field(default_factory=dict)


def _session_ok(row, cfg: SetupConfig) -> bool:
    if cfg.block_weekend and row.get("is_weekend", 0) == 1:
        return False
    if not cfg.allow_sessions:
        return True
    return any(row.get(s, 0) == 1 for s in cfg.allow_sessions)


def generate_setups(df: pd.DataFrame, symbol: str,
                    cfg: SetupConfig = None, with_features: bool = True) -> List[Setup]:
    """
    Return ORB+VWAP+trend+session setup events. `df` is OHLCV with 'timestamp';
    if with_features, intraday features are computed here (else assumed present).
    Causal: each setup uses only data up to its own bar.
    """
    cfg = cfg or SetupConfig()
    if with_features:
        df = build_intraday_features(df, cfg.or_minutes, cfg.bar_minutes)
    df = df.reset_index(drop=True)

    up_prev = df["or_broke_up"].shift(1).fillna(0)
    dn_prev = df["or_broke_down"].shift(1).fillna(0)
    # first bar that breaks the range (transition 0→1)
    new_up = (df["or_broke_up"] == 1) & (up_prev == 0)
    new_dn = (df["or_broke_down"] == 1) & (dn_prev == 0)

    setups: List[Setup] = []
    for i in df.index:
        if not (new_up.iloc[i] or new_dn.iloc[i]):
            continue
        row = df.iloc[i]
        atr = row.get("atr", np.nan)
        if not np.isfinite(atr) or atr <= 0:
            continue
        if not _session_ok(row, cfg):
            continue
        if row.get("vol_ratio", 0) < cfg.min_vol_ratio:
            continue

        if new_up.iloc[i]:
            direction = 1
            if cfg.require_trend and row.get("trend_up", 0) != 1:
                continue
            if cfg.require_vwap and row.get("above_vwap", 0) != 1:
                continue
        else:
            direction = -1
            if cfg.require_trend and row.get("trend_dn", 0) != 1:
                continue
            if cfg.require_vwap and row.get("above_vwap", 1) != 0:
                continue

        entry = float(row["close"])
        risk = cfg.stop_atr_mult * float(atr)
        if direction == 1:
            stop, target = entry - risk, entry + cfg.rr * risk
        else:
            stop, target = entry + risk, entry - cfg.rr * risk

        ctx = {c: float(row[c]) for c in CONTEXT_FEATURES if c in row and np.isfinite(row[c])}
        setups.append(Setup(ts=int(row["timestamp"]), symbol=symbol, direction=direction,
                            entry=entry, stop=stop, target=target, rr=cfg.rr,
                            bar_index=int(i), context=ctx))
    return setups


def simulate_setups(df: pd.DataFrame, setups: List[Setup],
                    cost_per_side: float = 0.0005, max_hold_bars: int = 100) -> List[dict]:
    """
    Walk each setup forward to its outcome (triple-barrier): does target hit
    before stop within max_hold_bars? Returns trade records with R-multiple and
    net %-return (after cost). One position at a time (non-overlapping).
    """
    df = df.reset_index(drop=True)
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    out: List[dict] = []
    busy_until = -1
    for s in setups:
        i = s.bar_index
        if i <= busy_until or i >= n - 1:
            continue
        d, entry, stop, target = s.direction, s.entry, s.stop, s.target
        risk = abs(entry - stop)
        exit_px, hit = None, "timeout"
        end = min(i + max_hold_bars + 1, n)
        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= stop:   exit_px, hit = stop, "stop"; break
                if high[j] >= target: exit_px, hit = target, "target"; break
            else:
                if high[j] >= stop:  exit_px, hit = stop, "stop"; break
                if low[j] <= target:  exit_px, hit = target, "target"; break
        if exit_px is None:
            j = min(j, n - 1); exit_px = close[j]
        gross = (exit_px - entry) / entry * d
        net = gross - 2 * cost_per_side
        r_multiple = ((exit_px - entry) * d) / risk if risk > 0 else 0.0
        # COST IN R-TERMS (the intraday killer): with fixed-fractional risk,
        # position size ∝ 1/stop_distance, so a tight stop makes round-trip cost
        # a LARGE fraction of the R risked. cost_r = 2·cost·entry/|entry-stop|.
        risk_frac = risk / entry if entry > 0 else 1.0
        cost_r = (2 * cost_per_side / risk_frac) if risk_frac > 0 else 0.0
        net_r = r_multiple - cost_r
        out.append({
            "ts": s.ts, "symbol": s.symbol, "direction": d, "bar_index": i,
            "entry": entry, "exit": exit_px, "hit": hit, "bars_held": j - i,
            "r_multiple": r_multiple, "net_r": net_r, "cost_r": cost_r,
            "net": net, "context": s.context,
        })
        busy_until = j
    return out
