"""
Replay report: per-strategy expectancy in R (net of costs), consistency over
chronological folds, R drawdown, and the approval rule inherited from the
intraday research (.claude/research/research_intraday.py):
  GO = n ≥ 30, mean R > 0, and positive in at least all-but-one of ≥ 3 folds.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from backend.allocator.thompson import p_positive

MIN_TRADES = 30
N_FOLDS = 4


def fold_consistency(rs: List[float], n_folds: int = N_FOLDS) -> dict:
    if len(rs) < n_folds * 5:
        return {"per_fold": [], "pos_folds": 0, "n_folds": 0}
    folds = [f for f in np.array_split(np.asarray(rs, dtype=float), n_folds) if len(f)]
    exps = [round(float(f.mean()), 4) for f in folds]
    return {"per_fold": exps, "pos_folds": sum(1 for e in exps if e > 0), "n_folds": len(exps)}


def max_drawdown_r(rs: List[float]) -> float:
    if not rs:
        return 0.0
    eq = np.cumsum(rs)
    return round(float((eq - np.maximum.accumulate(np.r_[0.0, eq])[1:]).min()), 4)


def strategy_report(trades: List[dict], days: float) -> dict:
    """`trades` of ONE strategy, chronological."""
    rs = [t["r"] for t in trades]
    n = len(rs)
    fc = fold_consistency(rs)
    exits = defaultdict(int)
    for t in trades:
        exits[t["exit_reason"]] += 1
    rep = {
        "n": n,
        "trades_per_day": round(n / max(days, 1e-9), 3),
        "mean_r": round(float(np.mean(rs)), 4) if n else None,
        "median_r": round(float(np.median(rs)), 4) if n else None,
        "win_rate": round(float(np.mean([r > 0 for r in rs])), 4) if n else None,
        "total_r": round(float(np.sum(rs)), 4),
        "max_dd_r": max_drawdown_r(rs),
        "p_positive": round(p_positive(rs), 4),
        "folds": fc,
        "exits": dict(exits),
        "long": _side(trades, 1), "short": _side(trades, -1),
    }
    rep["verdict"] = verdict(rep)
    return rep


def _side(trades: List[dict], d: int) -> dict:
    rs = [t["r"] for t in trades if t["direction"] == d]
    return {"n": len(rs), "mean_r": round(float(np.mean(rs)), 4) if rs else None}


def verdict(rep: dict) -> str:
    if rep["n"] < MIN_TRADES:
        return "INSUFFICIENT"
    fc = rep["folds"]
    consistent = fc["n_folds"] >= 3 and fc["pos_folds"] >= fc["n_folds"] - 1
    return "GO" if rep["mean_r"] > 0 and consistent else "NO-GO"


def report(trades: List[dict], days: float, names: Optional[List[str]] = None) -> Dict[str, dict]:
    by: Dict[str, List[dict]] = defaultdict(list)
    for t in sorted(trades, key=lambda t: t["bar_ts"]):
        by[t["hunter"]].append(t)
    for name in names or []:
        by.setdefault(name, [])
    return {name: strategy_report(ts, days) for name, ts in sorted(by.items())}
