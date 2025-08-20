from __future__ import annotations
from typing import List, Dict
import math


def calc_max_drawdown(equity_curve: List[Dict]) -> float:
    """Return max drawdown as a negative percentage (e.g., -12.3 for -12.3%)."""
    if not equity_curve:
        return 0.0
    peak = -math.inf
    max_dd = 0.0
    for p in equity_curve:
        eq = float(p.get("equity", 0.0))
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (eq / peak - 1.0) * 100.0
            if dd < max_dd:
                max_dd = dd
    return max_dd


def calc_sharpe(equity_curve: List[Dict], risk_free_rate: float = 0.0, period_per_year: int = 1440) -> float:
    """
    Approximate Sharpe ratio using simple returns between consecutive points.
    period_per_year is approximate for 1m data (60*24).
    """
    if len(equity_curve) < 2:
        return 0.0
    rets = []
    prev = float(equity_curve[0].get("equity", 0.0))
    for p in equity_curve[1:]:
        cur = float(p.get("equity", prev))
        if prev > 0:
            r = (cur / prev) - 1.0
            rets.append(r)
        prev = cur
    if not rets:
        return 0.0
    import statistics as stats

    mean_r = stats.mean(rets) - risk_free_rate / period_per_year
    std_r = stats.pstdev(rets) if len(rets) > 1 else 0.0
    if std_r == 0:
        return 0.0
    sharpe = (mean_r / std_r) * (period_per_year ** 0.5)
    return float(sharpe)
