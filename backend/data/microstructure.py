"""
Microstructure analysis — pure, testable functions over collected order book data.

Tests whether order book imbalance / microprice predict short-horizon mid-price
moves, and whether a threshold strategy (trade only on EXTREME imbalance) can pay
the cost. Data comes from the live collector (orderbook_data/*.csv).

Used by .claude/research/research_microstructure.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(mid: pd.Series, h: int) -> pd.Series:
    """Return over the next h snapshots: mid[t+h]/mid[t] - 1."""
    return mid.shift(-h) / mid - 1.0


def microprice_dev(df: pd.DataFrame) -> pd.Series:
    """(microprice - mid)/mid — the book's lean, a directional signal."""
    return (df["microprice"] - df["mid"]) / df["mid"]


def decay_correlations(df_sym: pd.DataFrame, horizons, signal_col: str = "imbalance_top20") -> dict:
    """Correlation of a signal at t with forward return at each horizon.
    df_sym must be one symbol, sorted by timestamp. Returns {h: corr}."""
    sig = microprice_dev(df_sym) if signal_col == "microprice_dev" else df_sym[signal_col]
    out = {}
    for h in horizons:
        fwd = forward_return(df_sym["mid"], h)
        v = pd.DataFrame({"s": sig.values, "f": fwd.values}).dropna()
        out[h] = float(v["s"].corr(v["f"])) if len(v) > 30 else float("nan")
    return out


def maker_viability(spread_bps: pd.Series, gross_bps: float,
                    taker_fee_bps_side: float = 2.0) -> dict:
    """
    Assess whether the h1 gross signal is capturable by any execution style.

    - taker floor = full spread + round-trip fees (2 * per-side fee): what a
      market order pays. Signal is viable as taker only if gross > this.
    - maker capture = half the spread you could EARN by posting passively. On a
      near-zero-spread name there is nothing to capture; on a wide-spread name
      you might, but you eat adverse selection (not modeled — so this is an
      OPTIMISTIC upper bound on the maker case).
    Returns the numbers so the caller can render an honest verdict.
    """
    med = float(spread_bps.median())
    taker_floor = med + 2.0 * taker_fee_bps_side
    return {
        "spread_med_bps": med,
        "half_spread_bps": med / 2.0,
        "taker_floor_bps": taker_floor,
        "gross_bps": float(gross_bps),
        "beats_taker": bool(gross_bps > taker_floor),
        "beats_maker_optimistic": bool(gross_bps > 0 and gross_bps > med / 2.0),
    }


def threshold_backtest(imbalance: pd.Series, fwd: pd.Series,
                       quantile: float, cost_bps: float) -> dict | None:
    """
    Trade only when |imbalance| >= its `quantile` (e.g. 0.9 = top 10% strongest).
    Direction = sign(imbalance). Net return per trade = sign*fwd - round-trip cost.
    Returns stats or None if too few trades.
    """
    v = pd.DataFrame({"imb": imbalance.values, "fwd": fwd.values}).dropna()
    if len(v) < 50:
        return None
    thr = v["imb"].abs().quantile(quantile)
    m = v[v["imb"].abs() >= thr]
    if len(m) < 10:
        return None
    sig = np.sign(m["imb"])
    gross = sig * m["fwd"]
    net = gross - cost_bps / 1e4          # round-trip cost as a return fraction
    return {
        "quantile": quantile, "cost_bps": cost_bps, "threshold": float(thr),
        "n": int(len(m)), "win_rate": float((gross > 0).mean()),
        "gross_bps": float(gross.mean() * 1e4), "net_bps": float(net.mean() * 1e4),
        "net_total_pct": float(net.sum() * 100),
    }
