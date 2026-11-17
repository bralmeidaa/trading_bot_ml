"""
What the student reads: compact, bucketed statistics — never raw trades
(fewer tokens, and nothing it could memorize trade by trade).

Two sources:
  - replay trades from the TRAINING segment only (the test segment that
    later validates its proposals is never shown);
  - the live shadow ledger (forward results, the judge's approve/veto record).
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np

from backend.hunters.strategies import BASE_HUNTERS
from backend.student.variants import FILTER_FEATURES, MAX_FILTERS, NOT_REPLAYABLE, OPS

MIN_BUCKET = 8            # smaller buckets are noise; not shown


def _hour_bucket(h) -> str:
    if h is None:
        return "?"
    h = int(h)
    return ("00-07 asia" if h < 7 else "07-13 london" if h < 13 else
            "13-16 london+ny" if h < 16 else "16-21 ny" if h < 21 else "21-24 late")


def _tercile(values: List[float]) -> Callable[[float], str]:
    if len(values) < 3:
        return lambda v: "all"
    lo, hi = np.quantile(values, [1 / 3, 2 / 3])
    return lambda v: "?" if v is None else (f"low(<{lo:.4g})" if v < lo
                                            else f"high(>{hi:.4g})" if v > hi else "mid")


def _stats(rs: List[float]) -> dict:
    return {"n": len(rs), "mean_r": round(float(np.mean(rs)), 3),
            "win_rate": round(float(np.mean([r > 0 for r in rs])), 3)}


def bucket_stats(trades: List[dict]) -> Dict[str, dict]:
    """trades: dicts with hunter, direction, r, context, exit_reason."""
    by_h: Dict[str, List[dict]] = defaultdict(list)
    for t in trades:
        by_h[t["hunter"]].append(t)
    out = {}
    for h, ts in sorted(by_h.items()):
        rs = [t["r"] for t in ts]
        ctx = [t.get("context") or {} for t in ts]
        splits: Dict[str, Callable[[dict, dict], str]] = {
            "direction": lambda t, c: "long" if t["direction"] == 1 else "short",
            "hour_utc": lambda t, c: _hour_bucket(c.get("hour")),
            "weekend": lambda t, c: "weekend" if c.get("is_weekend") else "weekday",
            "exit": lambda t, c: t.get("exit_reason", "?"),
        }
        for feat in ("atr15_pct", "vwap_dev", "ret_4h", "vol_ratio"):
            tf = _tercile([c[feat] for c in ctx if c.get(feat) is not None])
            splits[feat] = (lambda tf_, f: lambda t, c: tf_(c.get(f)))(tf, feat)
        buckets = {}
        for name, fn in splits.items():
            groups: Dict[str, List[float]] = defaultdict(list)
            for t, c in zip(ts, ctx):
                groups[fn(t, c)].append(t["r"])
            buckets[name] = {k: _stats(v) for k, v in sorted(groups.items()) if len(v) >= MIN_BUCKET}
        out[h] = {"overall": _stats(rs), "by": buckets}
    return out


def shadow_as_trades(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows:
        if r.get("r_multiple") is None:
            continue
        try:
            ctx = json.loads(r.get("context") or "{}")
        except (TypeError, ValueError):
            ctx = {}
        out.append({"hunter": r["hunter"], "direction": r["direction"], "r": r["r_multiple"],
                    "context": ctx, "exit_reason": r.get("exit_reason"),
                    "decision": (r.get("decision") or "").split(":")[0]})
    return out


def judge_record(shadow_trades: List[dict]) -> dict:
    groups: Dict[str, List[float]] = defaultdict(list)
    for t in shadow_trades:
        groups[t["decision"]].append(t["r"])
    return {k: _stats(v) for k, v in sorted(groups.items())}


def tunables() -> Dict[str, dict]:
    out = {}
    for name, factory in BASE_HUNTERS.items():
        h = factory()
        out[name] = {"doc": (type(h).__doc__ or "").strip().split("\n\n")[0][:400],
                     "replayable": name not in NOT_REPLAYABLE,
                     "params": {k: {"current": getattr(h, k), "min": lo, "max": hi}
                                for k, (lo, hi) in h.PARAMS.items()}}
    return out


def build_dossier(train_trades: List[dict], train_days: float, shadow_rows: List[dict],
                  variants: List[dict], lessons: List[dict]) -> dict:
    shadow = shadow_as_trades(shadow_rows)
    return {
        "costs": "every R is net of 0.15%/side; stops are >= 4x round-trip cost",
        "hunters": tunables(),
        "filter_language": {"features": FILTER_FEATURES, "ops": list(OPS),
                            "max_filters": MAX_FILTERS,
                            "example": {"feature": "hour", "op": "between", "value": [13, 20]}},
        "replay_training_segment": {"days": round(train_days, 1),
                                    "per_hunter": bucket_stats(train_trades)},
        "live_shadow_ledger": {"per_hunter": bucket_stats(shadow),
                               "judge_decisions": judge_record(shadow)},
        "existing_variants": [{"id": v["id"], "parent": v["parent"], "params": v["params"],
                               "filters": v["filters"], "status": v["status"],
                               "note": v.get("status_note")} for v in variants[-30:]],
        "active_lessons": [l["text"] for l in lessons if l.get("status") == "active"],
    }
