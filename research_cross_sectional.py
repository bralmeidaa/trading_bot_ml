# -*- coding: utf-8 -*-
"""
Cross-sectional strategy research harness (Phase 2 — the make-or-break gate).

Tests whether ranking a basket of majors by relative strength and trading
market-neutral (long top-k / short bottom-k) has out-of-sample, net-of-cost
edge. This is a DIFFERENT mechanism than single-pair prediction (which we
proved has no edge): it exploits RELATIVE moves, not absolute direction.

Walk-forward over time, costs charged on turnover, both momentum and reversal
ranking, optional BTC trend (regime) filter. Sweeps lookback × rebalance × k.

Run:
  python research_cross_sectional.py --timeframe 1h --days 365
  python research_cross_sectional.py --timeframe 4h --days 365
  python research_cross_sectional.py --timeframe 1d --days 1000
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
import numpy as np
import pandas as pd

from backend.data.universe import build_panel, DEFAULT_UNIVERSE, EXPANDED_UNIVERSE

# Cost per side as a fraction of notional traded (commission + slippage).
COST_PER_SIDE = 0.0015   # 0.15% — conservative for majors

# Bars per year by timeframe (for annualising Sharpe/return)
BARS_PER_YEAR = {"15m": 35040, "30m": 17520, "1h": 8760, "2h": 4380,
                 "4h": 2190, "1d": 365}


def cross_sectional_returns(close: pd.DataFrame, lookback: int, rebalance: int,
                            k: int, mode: str, btc_filter: bool,
                            cost: float = None,
                            volume: pd.DataFrame = None, max_universe: int = 15):
    """
    Simulate a market-neutral cross-sectional portfolio.

    Point-in-time eligible universe at each rebalance bar t:
      - symbol must have a non-NaN signal at t (i.e. it was listed >= lookback ago)
      - among those, keep the top `max_universe` by trailing quote-volume
        (close*volume mean over last ~30 bars) — mimics trading only what was
        actually liquid AT THAT DATE (survivorship-aware).
    Then rank within the eligible set:
      momentum: long top-k (winners), short bottom-k (losers); reversal = inverse.
      target weights +1/k on longs, -1/k on shorts (gross=2, net=0).
    Cost charged on turnover at each rebalance. Optional BTC regime filter.
    """
    if cost is None:
        cost = COST_PER_SIDE
    rets = close.pct_change().fillna(0.0)
    n, m = close.shape
    if m < 2 * k + 1 or n < lookback + rebalance + 5:
        return pd.Series(dtype=float)

    sig = close / close.shift(lookback) - 1.0
    if mode == "reversal":
        sig = -sig

    # trailing quote-volume (USD-ish liquidity proxy) for point-in-time filter
    if volume is not None:
        qvol = (close * volume).rolling(30, min_periods=5).mean()
    else:
        qvol = None

    regime_on = pd.Series(True, index=close.index)
    if btc_filter and "BTC/USDT" in close.columns:
        btc = close["BTC/USDT"]
        regime_on = (btc.ewm(span=max(2, lookback)).mean()
                     > btc.ewm(span=max(4, lookback * 5)).mean())

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    cur_w = pd.Series(0.0, index=close.columns)
    turnover = pd.Series(0.0, index=close.index)

    for t in range(lookback, n, rebalance):
        new_w = pd.Series(0.0, index=close.columns)
        if regime_on.iloc[t]:
            s = sig.iloc[t].dropna()
            # point-in-time liquidity cap: keep top max_universe by trailing qvol
            if qvol is not None and len(s) > max_universe:
                liq = qvol.iloc[t].reindex(s.index).dropna()
                if len(liq) >= 2 * k:
                    s = s.reindex(liq.sort_values().index[-max_universe:]).dropna()
            if len(s) >= 2 * k:
                ranked = s.sort_values()
                new_w[ranked.index[-k:]] = 1.0 / k     # longs (winners)
                new_w[ranked.index[:k]] = -1.0 / k     # shorts (losers)
        turnover.iloc[t] = (new_w - cur_w).abs().sum()
        cur_w = new_w
        weights.iloc[t:min(t + rebalance, n)] = new_w.values

    port = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)
    port = port - turnover * cost
    return port


def metrics(port: pd.Series, timeframe: str) -> dict:
    if port is None or len(port) == 0 or port.abs().sum() == 0:
        return dict(n=0, ann_ret=0, sharpe=0, maxdd=0, net=0, turnover=0)
    bpy = BARS_PER_YEAR.get(timeframe, 8760)
    mean, std = port.mean(), port.std()
    sharpe = (mean / std * np.sqrt(bpy)) if std > 0 else 0.0
    equity = (1 + port).cumprod()
    net = float(equity.iloc[-1] - 1)
    ann = float((1 + net) ** (bpy / len(port)) - 1) if len(port) > 0 else 0.0
    roll_max = equity.cummax()
    maxdd = float(((equity - roll_max) / roll_max).min())
    active = (port != 0).mean()
    return dict(n=len(port), ann_ret=ann, sharpe=float(sharpe),
                maxdd=maxdd, net=net, active=float(active))


def walk_forward(close: pd.DataFrame, timeframe: str, n_folds: int,
                 lookback: int, rebalance: int, k: int, mode: str, btc_filter: bool,
                 volume: pd.DataFrame = None, max_universe: int = 15):
    """Run the strategy on each contiguous OOS fold; return per-fold + aggregate metrics."""
    n = len(close)
    fold_size = n // (n_folds + 1)
    fold_sharpes, all_port = [], []
    for f in range(1, n_folds + 1):
        seg = close.iloc[f * fold_size:(f + 1) * fold_size]
        vseg = volume.iloc[f * fold_size:(f + 1) * fold_size] if volume is not None else None
        if len(seg) < lookback + rebalance + 5:
            continue
        port = cross_sectional_returns(seg, lookback, rebalance, k, mode, btc_filter,
                                       volume=vseg, max_universe=max_universe)
        if len(port):
            mtr = metrics(port, timeframe)
            fold_sharpes.append(mtr["sharpe"])
            all_port.append(port)
    if not all_port:
        return None
    agg = metrics(pd.concat(all_port), timeframe)
    agg["fold_sharpes"] = fold_sharpes
    agg["pos_folds"] = sum(1 for s in fold_sharpes if s > 0)
    agg["n_folds"] = len(fold_sharpes)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--pit", action="store_true",
                    help="survivorship-aware: expanded universe + point-in-time eligibility")
    ap.add_argument("--max-universe", type=int, default=15,
                    help="liquidity cap: trade only top-N by trailing volume per bar")
    ap.add_argument("--cost", type=float, default=None,
                    help="override cost per side (fraction, e.g. 0.0025) for sensitivity")
    args = ap.parse_args()

    global COST_PER_SIDE
    if args.cost is not None:
        COST_PER_SIDE = args.cost

    print("=" * 78)
    mode_lbl = "PIT (survivorship-aware)" if args.pit else "dense (survivorship-biased)"
    print(f"CROSS-SECTIONAL RESEARCH — {args.timeframe}, {args.days}d, {args.folds} folds, "
          f"cost {COST_PER_SIDE*100:.2f}%/side | {mode_lbl}")
    print("=" * 78)
    uni = EXPANDED_UNIVERSE if args.pit else DEFAULT_UNIVERSE
    close, volume, kept = build_panel(symbols=uni, timeframe=args.timeframe,
                                      days=args.days, point_in_time=args.pit)
    if close.shape[1] < 6:
        print("Universe too small — aborting.")
        return

    # parameter grid (lookback & rebalance in BARS for this timeframe)
    lookbacks  = [12, 24, 48, 96]
    rebalances = [6, 12, 24]
    ks         = [3, 4]
    modes      = ["momentum", "reversal"]

    rows = []
    for mode in modes:
        for rf in (False, True):
            for lb in lookbacks:
                for rb in rebalances:
                    for k in ks:
                        agg = walk_forward(close, args.timeframe, args.folds,
                                           lb, rb, k, mode, rf,
                                           volume=volume, max_universe=args.max_universe)
                        if agg:
                            rows.append((mode, rf, lb, rb, k, agg))

    rows.sort(key=lambda r: r[5]["sharpe"], reverse=True)

    print(f"\n{'mode':<10}{'regime':<7}{'LB':>4}{'RB':>4}{'k':>3} | "
          f"{'Sharpe':>7}{'annRet':>9}{'maxDD':>8}{'pos/folds':>10}")
    print("-" * 78)
    for mode, rf, lb, rb, k, a in rows[:15]:
        print(f"{mode:<10}{('on' if rf else 'off'):<7}{lb:>4}{rb:>4}{k:>3} | "
              f"{a['sharpe']:>7.2f}{a['ann_ret']:>+8.1%}{a['maxdd']:>+8.1%}"
              f"{a['pos_folds']:>5}/{a['n_folds']:<4}")

    print("\n" + "=" * 78)
    if rows:
        m, rf, lb, rb, k, a = rows[0]
        go = a["sharpe"] > 1.0 and a["pos_folds"] >= max(1, a["n_folds"] - 1) and a["ann_ret"] > 0
        print(f"MELHOR: {m}, regime={'on' if rf else 'off'}, LB={lb}, RB={rb}, k={k}")
        print(f"  Sharpe={a['sharpe']:.2f}  annRet={a['ann_ret']:+.1%}  "
              f"maxDD={a['maxdd']:+.1%}  folds+={a['pos_folds']}/{a['n_folds']}")
        print(f"\n  VEREDITO: {'GO' if go else 'NO-GO'} — "
              f"{'edge consistente OOS, prosseguir p/ Fase 3' if go else 'sem edge robusto neste timeframe'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
