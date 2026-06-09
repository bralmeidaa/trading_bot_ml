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

from backend.data.universe import build_panel

# Cost per side as a fraction of notional traded (commission + slippage).
COST_PER_SIDE = 0.0015   # 0.15% — conservative for majors

# Bars per year by timeframe (for annualising Sharpe/return)
BARS_PER_YEAR = {"15m": 35040, "30m": 17520, "1h": 8760, "2h": 4380,
                 "4h": 2190, "1d": 365}


def cross_sectional_returns(close: pd.DataFrame, lookback: int, rebalance: int,
                            k: int, mode: str, btc_filter: bool,
                            cost: float = COST_PER_SIDE):
    """
    Simulate a market-neutral cross-sectional portfolio.

    At each rebalance bar t:
      - rank symbols by signal over `lookback` bars
        mode='momentum': signal = return over lookback (long winners, short losers)
        mode='reversal': signal = -return over lookback (long losers, short winners)
      - target weights: +1/k on top-k, -1/k on bottom-k (gross=2, net=0)
      - hold until next rebalance; portfolio return = sum(w * forward bar returns)
      - charge `cost` on the turnover (sum of |w_new - w_old|) at each rebalance
      - optional BTC regime filter: only hold positions when BTC > its slow EMA

    Returns a pd.Series of per-bar net portfolio returns indexed by close.index.
    """
    rets = close.pct_change().fillna(0.0)
    n, m = close.shape
    if m < 2 * k + 1 or n < lookback + rebalance + 5:
        return pd.Series(dtype=float)

    # signal = trailing return over lookback
    sig = close / close.shift(lookback) - 1.0
    if mode == "reversal":
        sig = -sig

    # BTC regime: trade only when BTC above its slow EMA (trend-on)
    regime_on = pd.Series(True, index=close.index)
    if btc_filter and "BTC/USDT" in close.columns:
        btc = close["BTC/USDT"]
        ema_fast = btc.ewm(span=max(2, lookback)).mean()
        ema_slow = btc.ewm(span=max(4, lookback * 5)).mean()
        regime_on = (ema_fast > ema_slow)

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    cur_w = pd.Series(0.0, index=close.columns)
    turnover = pd.Series(0.0, index=close.index)

    rebal_idx = range(lookback, n, rebalance)
    for t in rebal_idx:
        ts = close.index[t]
        if not regime_on.iloc[t]:
            new_w = pd.Series(0.0, index=close.columns)
        else:
            s = sig.iloc[t].dropna()
            if len(s) < 2 * k:
                new_w = pd.Series(0.0, index=close.columns)
            else:
                ranked = s.sort_values()
                longs = ranked.index[-k:]
                shorts = ranked.index[:k]
                new_w = pd.Series(0.0, index=close.columns)
                new_w[longs] = 1.0 / k
                new_w[shorts] = -1.0 / k
        turnover.iloc[t] = (new_w - cur_w).abs().sum()
        cur_w = new_w
        # apply weights until next rebalance
        end = min(t + rebalance, n)
        weights.iloc[t:end] = new_w.values

    # portfolio return per bar = sum_i w_i(t-1) * ret_i(t)
    port = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)
    # subtract cost on turnover at each rebalance bar
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
                 lookback: int, rebalance: int, k: int, mode: str, btc_filter: bool):
    """Run the strategy on each contiguous OOS fold; return per-fold + aggregate metrics."""
    n = len(close)
    fold_size = n // (n_folds + 1)
    fold_sharpes, all_port = [], []
    for f in range(1, n_folds + 1):
        seg = close.iloc[f * fold_size:(f + 1) * fold_size]
        if len(seg) < lookback + rebalance + 5:
            continue
        port = cross_sectional_returns(seg, lookback, rebalance, k, mode, btc_filter)
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
    args = ap.parse_args()

    print("=" * 78)
    print(f"CROSS-SECTIONAL RESEARCH — {args.timeframe}, {args.days}d, {args.folds} folds, "
          f"cost {COST_PER_SIDE*100:.2f}%/side")
    print("=" * 78)
    close, volume, kept = build_panel(timeframe=args.timeframe, days=args.days)
    if close.shape[1] < 6:
        print("Universe too small after coverage filter — aborting.")
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
                                           lb, rb, k, mode, rf)
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
