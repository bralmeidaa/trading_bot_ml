"""
Replay simulator: runs the live hunters over historical 5m bars, bar by bar.

Same code path as live — `features.prepare` + `Hunter.scan` on a trailing
window as long as the live engine keeps (600 bars) — with execution rules
that are, if anything, harsher than live:
  - the signal is the CLOSE of bar i; entry is the OPEN of bar i+1;
  - 0.15% per side (commission + slippage, the paper broker's own fills);
  - stop/target levels from the filled entry, like the shadow ledger;
  - stop is checked before target inside a bar (conservative), and a bar
    that OPENS through a level fills at that open (gaps hurt, never help);
  - time stop after `max_hold_bars` at that bar's close.

Like the live shadow ledger, every candidate is followed independently (no
book-level gates), with the same per hunter+symbol cooldown. The LLM judge is
never replayed: asking it about the past would leak hindsight; its value is
measured live.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from backend.agent.paper_execution import ROUND_TRIP_COST, fill_entry_price, fill_exit_price
from backend.hunters.base import Candidate, Hunter, ScanContext
from backend.hunters.features import prepare

logger = logging.getLogger(__name__)

BAR_MS = 5 * 60 * 1000


@dataclass
class ReplayConfig:
    window: int = 600                    # bars each scan sees (live: HunterConfig.bar_history)
    warmup_bars: int = 300               # skip the first bars: features still forming
    cooldown_bars: int = 12              # same hunter+symbol at most once an hour
    max_hold_bars: int = 288             # 24h
    min_stop_pct: float = 4 * ROUND_TRIP_COST
    max_stop_pct: float = 0.08


def simulate_trade(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, i: int,
                   direction: int, stop_pct: float, target_pct: float,
                   max_hold_bars: int) -> Optional[dict]:
    """Outcome of a signal at the close of bar i. None if the data ends first."""
    n = len(c)
    if i + 1 >= n:
        return None
    d = direction
    entry = fill_entry_price(o[i + 1], d)
    stop = entry * (1 - d * stop_pct)
    target = entry * (1 + d * target_pct)
    for j in range(i + 1, min(i + 1 + max_hold_bars, n)):
        if d == 1:
            if o[j] <= stop:      raw, why = o[j], "stop"
            elif l[j] <= stop:    raw, why = stop, "stop"
            elif o[j] >= target:  raw, why = o[j], "target"
            elif h[j] >= target:  raw, why = target, "target"
            else:                 continue
        else:
            if o[j] >= stop:      raw, why = o[j], "stop"
            elif h[j] >= stop:    raw, why = stop, "stop"
            elif o[j] <= target:  raw, why = o[j], "target"
            elif l[j] <= target:  raw, why = target, "target"
            else:                 continue
        break
    else:
        j = i + max_hold_bars
        if j >= n:
            return None
        raw, why = c[j], "time"
    exit_px = fill_exit_price(raw, d)
    pnl_pct = (exit_px - entry) / entry * d
    return {"entry": entry, "exit": exit_px, "exit_reason": why, "bars_held": j - i,
            "exit_bar": j, "pnl_pct": pnl_pct, "r": pnl_pct / stop_pct}


def _candidates(h: Hunter, symbol: str, df: pd.DataFrame, ctx: ScanContext,
                cfg: ReplayConfig, start_idx: int) -> Iterable[tuple]:
    """(bar index, Candidate) for every bar >= start_idx where `h` fires."""
    batch = h.scan_batch(symbol, df, ctx)
    if batch is not None:
        pos = {int(t): k for k, t in enumerate(df["timestamp"].to_numpy())}
        for cand in batch:
            k = pos.get(cand.bar_ts)
            if k is not None and k >= start_idx:
                yield k, cand
        return
    mask = h.prefilter(df)
    idx = np.arange(len(df)) if mask is None else np.flatnonzero(np.asarray(mask, dtype=bool))
    for k in idx[idx >= start_idx]:
        cand = h.scan(symbol, df.iloc[max(0, k - cfg.window + 1):k + 1], ctx)
        if cand is not None:
            yield int(k), cand


def replay_symbol(symbol: str, bars: pd.DataFrame, hunters: List[Hunter],
                  cfg: Optional[ReplayConfig] = None, start_ms: Optional[int] = None,
                  end_ms: Optional[int] = None) -> List[dict]:
    """All simulated trades of `hunters` on one symbol. Signals are taken only
    from bars with open time in [start_ms, end_ms) — bars before start_ms are
    still used as feature history — and exits may run past end_ms."""
    cfg = cfg or ReplayConfig()
    if len(bars) < cfg.warmup_bars + 2:
        return []
    df = prepare(bars)
    ts = df["timestamp"].to_numpy()
    o, hi, lo, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    start_idx = cfg.warmup_bars
    if start_ms is not None:
        start_idx = max(start_idx, int(np.searchsorted(ts, start_ms)))
    end_idx = len(df) if end_ms is None else int(np.searchsorted(ts, end_ms))
    ctx = ScanContext(min_stop_pct=cfg.min_stop_pct, max_stop_pct=cfg.max_stop_pct)
    trades: List[dict] = []
    for h in hunters:
        last_k = -10 ** 9
        try:
            for k, cand in _candidates(h, symbol, df, ctx, cfg, start_idx):
                if k >= end_idx or k - last_k < cfg.cooldown_bars:
                    continue
                res = simulate_trade(o, hi, lo, c, k, cand.direction, cand.stop_pct,
                                     cand.target_pct, cfg.max_hold_bars)
                if res is None:
                    continue
                last_k = k
                trades.append({"hunter": cand.hunter, "version": cand.version, "symbol": symbol,
                               "direction": cand.direction, "bar_ts": int(ts[k]),
                               "exit_ts": int(ts[res["exit_bar"]]), "stop_pct": cand.stop_pct,
                               "target_pct": cand.target_pct, "context": cand.context,
                               **{k2: v for k2, v in res.items() if k2 != "exit_bar"}})
        except Exception as exc:          # one broken hunter must not sink the run
            logger.warning(f"replay: {h.name} failed on {symbol}: {exc}")
    return trades


def replay(bars_by_symbol: Dict[str, pd.DataFrame], hunters: List[Hunter],
           cfg: Optional[ReplayConfig] = None, start_ms: Optional[int] = None,
           end_ms: Optional[int] = None) -> List[dict]:
    out: List[dict] = []
    for sym, bars in bars_by_symbol.items():
        out.extend(replay_symbol(sym, bars, hunters, cfg, start_ms, end_ms))
    out.sort(key=lambda t: t["bar_ts"])
    return out
