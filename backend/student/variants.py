"""
What the LLM student is allowed to propose, and how a proposal becomes a
hunter.

A variant = a seeded hunter + parameter changes (each within the bounds the
hunter declares in PARAMS) + up to 3 filters over a CLOSED list of bar
features. No code is generated: the spec is data, validated here, and the
variant runs the parent's own `scan`.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple

from backend.hunters.base import Candidate, Hunter
from backend.hunters.strategies import BASE_HUNTERS

# feature -> plausible value range (a filter value outside it is a typo, not a hypothesis)
FILTER_FEATURES: Dict[str, Tuple[float, float]] = {
    "hour": (0, 23), "dow": (0, 6), "is_weekend": (0, 1),
    "sess_asia": (0, 1), "sess_london": (0, 1), "sess_ny": (0, 1),
    "vwap_dev": (-0.3, 0.3), "atr15_pct": (0.0, 0.1),
    "ret_1h": (-0.5, 0.5), "ret_4h": (-0.5, 0.5), "vol_ratio": (0.0, 50.0),
    "direction": (-1, 1),
}
OPS = ("<", ">", "between", "in")
MAX_FILTERS = 3
# replay can't judge these: no historical liquidation feed
NOT_REPLAYABLE = {"liquidation_reversal"}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_spec(spec: dict) -> Tuple[Optional[dict], Optional[str]]:
    """Normalized spec {parent, params, filters, rationale} or (None, error)."""
    if not isinstance(spec, dict):
        return None, "spec must be an object"
    parent = spec.get("parent")
    if parent not in BASE_HUNTERS:
        return None, f"unknown parent {parent!r}"
    if parent in NOT_REPLAYABLE:
        return None, f"{parent} cannot be validated in replay (no historical liquidations)"
    base = BASE_HUNTERS[parent]()
    params = spec.get("params") or {}
    if not isinstance(params, dict):
        return None, "params must be an object"
    clean_params = {}
    for k, v in params.items():
        if k not in base.PARAMS:
            return None, f"{parent} has no tunable parameter {k!r} (allowed: {sorted(base.PARAMS)})"
        lo, hi = base.PARAMS[k]
        if not _num(v) or not lo <= v <= hi:
            return None, f"{k}={v!r} outside [{lo}, {hi}]"
        cast = type(getattr(base, k))(v)
        if cast != getattr(base, k):
            clean_params[k] = cast
    clean_filters, err = validate_filters(spec.get("filters") or [])
    if err:
        return None, err
    if not clean_params and not clean_filters:
        return None, "no change versus the parent"
    return {"parent": parent, "params": clean_params, "filters": clean_filters,
            "rationale": str(spec.get("rationale", ""))[:1000]}, None


def validate_filters(filters) -> Tuple[Optional[List[dict]], Optional[str]]:
    if not isinstance(filters, list) or len(filters) > MAX_FILTERS:
        return None, f"filters must be a list of at most {MAX_FILTERS}"
    clean_filters = []
    for f in filters:
        if not isinstance(f, dict):
            return None, "each filter must be an object"
        feat, op, val = f.get("feature"), f.get("op"), f.get("value")
        if feat not in FILTER_FEATURES:
            return None, f"feature {feat!r} not in the allowed list {sorted(FILTER_FEATURES)}"
        if op not in OPS:
            return None, f"op {op!r} not in {OPS}"
        lo, hi = FILTER_FEATURES[feat]
        vals = val if op in ("between", "in") else [val]
        if not isinstance(vals, list) or not vals or not all(_num(x) for x in vals):
            return None, f"bad value for {feat} {op}: {val!r}"
        if op == "between" and (len(vals) != 2 or vals[0] > vals[1]):
            return None, "between needs [low, high]"
        if any(not lo <= x <= hi for x in vals):
            return None, f"{feat} value outside its range [{lo}, {hi}]"
        clean_filters.append({"feature": feat, "op": op, "value": val})
    return clean_filters, None


def spec_hash(spec: dict) -> str:
    key = json.dumps({"parent": spec["parent"], "params": spec["params"],
                      "filters": sorted(spec["filters"], key=lambda f: json.dumps(f, sort_keys=True))},
                     sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def passes(filters: List[dict], context: dict, direction: int) -> bool:
    """All filters hold. A feature missing from the context fails the filter
    (fail closed)."""
    for f in filters:
        feat, op, val = f["feature"], f["op"], f["value"]
        x = direction if feat == "direction" else context.get(feat)
        if x is None:
            return False
        if op == "<" and not x < val:
            return False
        if op == ">" and not x > val:
            return False
        if op == "between" and not val[0] <= x <= val[1]:
            return False
        if op == "in" and x not in val:
            return False
    return True


class VariantHunter(Hunter):
    """Runs the parent's scan with the variant's parameters, keeps only
    candidates that pass the filters, and labels them with its own name."""
    PARAMS = {}

    def __init__(self, variant_id: str, spec: dict):
        self.base = BASE_HUNTERS[spec["parent"]]().with_params(**spec["params"])
        self.name = variant_id
        self.version = variant_id.rsplit("~", 1)[-1]
        self.parent = spec["parent"]
        self.live = True
        self.spec = spec

    def _relabel(self, c: Optional[Candidate]) -> Optional[Candidate]:
        if c is None or not passes(self.spec["filters"], c.context, c.direction):
            return None
        c.hunter, c.version = self.name, self.version
        c.rationale = f"[variant of {self.parent}] {c.rationale}"
        return c

    def prefilter(self, df):
        return self.base.prefilter(df)

    def scan_batch(self, symbol, df, ctx):
        batch = self.base.scan_batch(symbol, df, ctx)
        return None if batch is None else [c for c in map(self._relabel, batch) if c]

    def scan(self, symbol, df, ctx):
        return self._relabel(self.base.scan(symbol, df, ctx))


def build_variant(row: dict) -> VariantHunter:
    """From a strategy_variants row (params/filters already decoded)."""
    return VariantHunter(row["id"], {"parent": row["parent"], "params": row["params"] or {},
                                     "filters": row["filters"] or []})
