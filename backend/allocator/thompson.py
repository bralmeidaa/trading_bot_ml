"""
Strategy tournament allocator.

Each strategy's net R per trade is modelled as Normal with unknown mean and
variance (Normal-Inverse-Gamma conjugate prior centred on ZERO edge — a new
strategy is assumed worthless until its trades say otherwise). From the
posterior we get:
  - p_positive: P(mean R > 0);
  - a Thompson-sampling weight: how often the strategy has the best sampled
    mean among the eligible ones, with an always-present "cash" arm at 0R —
    so when nothing is convincingly positive, most weight sits in cash.

Rules (paper money, but the same rules would hold for real money):
  - fewer than `min_trades` closed forward trades → "warming": a seeded
    strategy keeps trading at `explore_scale` unless the replay already
    rejected it (NO-GO); a student variant gets nothing (shadow only);
  - after that, below `demote_below` p_positive → "demoted" (shadow);
    otherwise risk scale = rank × confidence, where rank = Thompson weight /
    the best strategy's weight and confidence ramps linearly from 0 at
    `demote_below` to 1 at `full_confidence_at` p_positive. Full risk needs
    both: leading the field AND P(edge > 0) ≥ 0.9;
  - shadow-only controls never get capital.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy import stats

# prior: mean 0 with the weight of `kappa0` trades; variance ~ beta0/(alpha0-1) = 1R²
MU0, KAPPA0, ALPHA0, BETA0 = 0.0, 1.0, 2.0, 1.0
CASH = "_cash"


def posterior(rs: Iterable[float], mu0=MU0, kappa0=KAPPA0, alpha0=ALPHA0, beta0=BETA0):
    x = np.asarray(list(rs), dtype=float)
    n = len(x)
    if n == 0:
        return mu0, kappa0, alpha0, beta0
    mean = float(x.mean())
    ss = float(((x - mean) ** 2).sum())
    kappa = kappa0 + n
    mu = (kappa0 * mu0 + n * mean) / kappa
    alpha = alpha0 + n / 2
    beta = beta0 + 0.5 * ss + kappa0 * n * (mean - mu0) ** 2 / (2 * kappa)
    return mu, kappa, alpha, beta


def mean_marginal(rs: Iterable[float]):
    """Student-t marginal of the mean: (df, loc, scale)."""
    mu, kappa, alpha, beta = posterior(rs)
    return 2 * alpha, mu, float(np.sqrt(beta / (alpha * kappa)))


def p_positive(rs: Iterable[float]) -> float:
    df, loc, scale = mean_marginal(rs)
    return float(stats.t.sf(0.0, df, loc=loc, scale=scale))


def thompson_weights(samples_by_name: Dict[str, List[float]], draws: int = 4000,
                     seed: Optional[int] = 0) -> Dict[str, float]:
    """Share of posterior draws in which each strategy has the highest mean,
    competing against a cash arm fixed at 0R (returned as CASH). Weights sum
    to 1."""
    names = list(samples_by_name)
    if not names:
        return {CASH: 1.0}
    rng = np.random.default_rng(seed)
    cols = []
    for name in names:
        df, loc, scale = mean_marginal(samples_by_name[name])
        cols.append(loc + scale * rng.standard_t(df, size=draws))
    cols.append(np.zeros(draws))                       # cash
    best = np.argmax(np.vstack(cols), axis=0)
    return {name: float((best == k).mean()) for k, name in enumerate(names + [CASH])}


@dataclass
class AllocatorConfig:
    min_trades: int = 30
    explore_scale: float = 0.5
    demote_below: float = 0.5
    full_confidence_at: float = 0.9


def summarize(rs: List[float]) -> dict:
    n = len(rs)
    return {"n": n, "mean_r": round(float(np.mean(rs)), 4) if n else None,
            "win_rate": round(float(np.mean([r > 0 for r in rs])), 4) if n else None,
            "total_r": round(float(np.sum(rs)), 4) if n else 0.0,
            "p_positive": round(p_positive(rs), 4)}


def allocate(r_by_strategy: Dict[str, List[float]], meta: Dict[str, dict],
             cfg: Optional[AllocatorConfig] = None) -> Dict[str, dict]:
    """`meta[name]`: {"live": bool, "variant": bool, "replay": verdict|None}. Returns, per strategy,
    its stats, status (control / warming / active / demoted), Thompson weight
    and the risk scale the live engine applies (0 = shadow only)."""
    cfg = cfg or AllocatorConfig()
    out: Dict[str, dict] = {}
    eligible: Dict[str, List[float]] = {}
    for name, m in meta.items():
        rs = list(r_by_strategy.get(name, []))
        row = summarize(rs)
        if not m.get("live", True):
            row.update(status="control", risk_scale=0.0)
        elif row["n"] < cfg.min_trades:
            funded = not m.get("variant") and m.get("replay") != "NO-GO"
            row.update(status="warming", risk_scale=cfg.explore_scale if funded else 0.0)
        elif row["p_positive"] < cfg.demote_below:
            row.update(status="demoted", risk_scale=0.0)
        else:
            row.update(status="active")
            eligible[name] = rs
        if m.get("replay"):
            row["replay"] = m["replay"]
        out[name] = row
    weights = thompson_weights(eligible)
    top = max((w for n, w in weights.items() if n != CASH), default=0.0)
    span = max(cfg.full_confidence_at - cfg.demote_below, 1e-9)
    for name, row in out.items():
        w = weights.get(name, 0.0)
        row["weight"] = round(w, 4)
        if row["status"] == "active":
            conf = min(1.0, max(0.0, (row["p_positive"] - cfg.demote_below) / span))
            row["risk_scale"] = round((w / top if top > 0 else 0.0) * conf, 4)
    return out


def cash_weight(allocation: Dict[str, dict]) -> float:
    return round(max(0.0, 1.0 - sum(r.get("weight", 0.0) for r in allocation.values())), 4)
