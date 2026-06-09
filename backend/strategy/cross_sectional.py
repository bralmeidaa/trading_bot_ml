"""
Cross-sectional momentum strategy — pure, testable core.

Validated (docs/STRATEGY_THESIS.md): daily, market-neutral long top-k / short
bottom-k by relative momentum, BTC-EMA regime filter, point-in-time liquidity
universe. GO out-of-sample (Sharpe ~1.87, 4/4 folds, survivorship-aware,
cost-robust to 0.40%/side).

This module holds the pure functions (no I/O) so they can be unit-tested and
reused by the research harness and (Phase 3) the live portfolio engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_COST_PER_SIDE = 0.0015   # 0.15% commission+slippage per side
BARS_PER_YEAR = {"15m": 35040, "30m": 17520, "1h": 8760, "2h": 4380,
                 "4h": 2190, "1d": 365}


def target_weights(sig_row: pd.Series, k: int,
                   liquidity_row: pd.Series = None, max_universe: int = 15) -> pd.Series:
    """
    Given a signal row (one value per symbol, NaN = not eligible), return the
    market-neutral target weights: +1/k on the top-k, -1/k on the bottom-k.

    Point-in-time liquidity cap: if liquidity_row given and the eligible set is
    larger than max_universe, keep only the top max_universe by liquidity before
    ranking — mimics trading only what was liquid at that date.
    """
    w = pd.Series(0.0, index=sig_row.index)
    s = sig_row.dropna()
    if liquidity_row is not None and len(s) > max_universe:
        liq = liquidity_row.reindex(s.index).dropna()
        if len(liq) >= 2 * k:
            s = s.reindex(liq.sort_values().index[-max_universe:]).dropna()
    if len(s) < 2 * k:
        return w
    ranked = s.sort_values()
    w[ranked.index[-k:]] = 1.0 / k     # longs = winners (highest signal)
    w[ranked.index[:k]] = -1.0 / k     # shorts = losers (lowest signal)
    return w


def momentum_signal(close: pd.DataFrame, lookback: int, mode: str = "momentum") -> pd.DataFrame:
    """Trailing return over `lookback`. reversal mode negates it."""
    sig = close / close.shift(lookback) - 1.0
    return -sig if mode == "reversal" else sig


def btc_regime(close: pd.DataFrame, lookback: int) -> pd.Series:
    """Trend-on when BTC fast EMA > slow EMA. All-True if BTC absent."""
    if "BTC/USDT" not in close.columns:
        return pd.Series(True, index=close.index)
    btc = close["BTC/USDT"]
    return btc.ewm(span=max(2, lookback)).mean() > btc.ewm(span=max(4, lookback * 5)).mean()


def simulate(close: pd.DataFrame, lookback: int, rebalance: int, k: int,
             mode: str = "momentum", btc_filter: bool = True,
             cost: float = DEFAULT_COST_PER_SIDE, volume: pd.DataFrame = None,
             max_universe: int = 15) -> pd.Series:
    """
    Simulate the market-neutral cross-sectional portfolio. Returns per-bar net
    return series. No look-ahead: weights set at rebalance bar t use signal at t,
    portfolio return at t+1 uses weights from t (shift).
    """
    rets = close.pct_change().fillna(0.0)
    n, m = close.shape
    if m < 2 * k + 1 or n < lookback + rebalance + 5:
        return pd.Series(dtype=float)

    sig = momentum_signal(close, lookback, mode)
    qvol = (close * volume).rolling(30, min_periods=5).mean() if volume is not None else None
    regime_on = btc_regime(close, lookback) if btc_filter else pd.Series(True, index=close.index)

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    cur_w = pd.Series(0.0, index=close.columns)
    turnover = pd.Series(0.0, index=close.index)

    for t in range(lookback, n, rebalance):
        if regime_on.iloc[t]:
            liq = qvol.iloc[t] if qvol is not None else None
            new_w = target_weights(sig.iloc[t], k, liq, max_universe)
        else:
            new_w = pd.Series(0.0, index=close.columns)
        turnover.iloc[t] = (new_w - cur_w).abs().sum()
        cur_w = new_w
        weights.iloc[t:min(t + rebalance, n)] = new_w.values

    port = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)
    return port - turnover * cost


def metrics(port: pd.Series, timeframe: str = "1d") -> dict:
    """Performance metrics from a per-bar return series."""
    if port is None or len(port) == 0 or port.abs().sum() == 0:
        return dict(n=0, ann_ret=0.0, sharpe=0.0, maxdd=0.0, net=0.0)
    bpy = BARS_PER_YEAR.get(timeframe, 8760)
    mean, std = port.mean(), port.std()
    sharpe = float(mean / std * np.sqrt(bpy)) if std > 0 else 0.0
    equity = (1 + port).cumprod()
    net = float(equity.iloc[-1] - 1)
    ann = float((1 + net) ** (bpy / len(port)) - 1)
    roll_max = equity.cummax()
    maxdd = float(((equity - roll_max) / roll_max).min())
    return dict(n=len(port), ann_ret=ann, sharpe=sharpe, maxdd=maxdd, net=net)


def walk_forward(close: pd.DataFrame, timeframe: str, n_folds: int, lookback: int,
                 rebalance: int, k: int, mode: str = "momentum", btc_filter: bool = True,
                 cost: float = DEFAULT_COST_PER_SIDE, volume: pd.DataFrame = None,
                 max_universe: int = 15) -> dict | None:
    """Out-of-sample walk-forward over contiguous folds; aggregate + per-fold Sharpe."""
    n = len(close)
    fold = n // (n_folds + 1)
    sharpes, ports = [], []
    for f in range(1, n_folds + 1):
        seg = close.iloc[f * fold:(f + 1) * fold]
        vseg = volume.iloc[f * fold:(f + 1) * fold] if volume is not None else None
        if len(seg) < lookback + rebalance + 5:
            continue
        p = simulate(seg, lookback, rebalance, k, mode, btc_filter, cost, vseg, max_universe)
        if len(p):
            sharpes.append(metrics(p, timeframe)["sharpe"])
            ports.append(p)
    if not ports:
        return None
    agg = metrics(pd.concat(ports), timeframe)
    agg["fold_sharpes"] = sharpes
    agg["pos_folds"] = sum(1 for s in sharpes if s > 0)
    agg["n_folds"] = len(sharpes)
    return agg
