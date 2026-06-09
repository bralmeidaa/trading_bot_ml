# -*- coding: utf-8 -*-
"""
Tuning harness for LINK/USDT — sweep the decision gates to raise trade
frequency while keeping a good win rate. Uses the SAME live decision methods
(_check_* + _combine_signals) so results predict live behaviour.

Speed trick: the ML model is trained ONCE per walk-forward fold; the tuning
parameters (min_vote, confidence_threshold) only affect decision-time, so all
configs reuse the same trained model + precomputed sub-signals.

Run:  python tune_link.py --days 365 --splits 4
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import time
import argparse
from datetime import datetime, timedelta

import ccxt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

from production_trading_system import OptimizedSignalGenerator

SYMBOL, TIMEFRAME = "LINK/USDT", "5m"
COMMISSION, SLIPPAGE = 0.001, 0.0005
ATR_STOP_MULT, ATR_TP_MULT = 1.5, 2.5
MAX_HOLD_BARS = 200

# Config grid: (min_vote, confidence_threshold)
MIN_VOTES   = [0.30, 0.20, 0.15, 0.10]
CONF_THRS   = [0.65, 0.55]


def fetch(symbol, timeframe, days):
    ex = ccxt.binance({"enableRateLimit": True, "rateLimit": 1200})
    tf_min = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}.get(timeframe, 5)
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    out = []
    print(f"  downloading {symbol} {timeframe} {days}d...", end="", flush=True)
    while True:
        b = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not b:
            break
        out.extend(b)
        since = b[-1][0] + tf_min * 60 * 1000
        if len(b) < 1000:
            break
        time.sleep(0.25)
    df = pd.DataFrame(out, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    print(f" {len(df):,} candles")
    return df


def train_fold(gen, train_df):
    """Fast single-fit training (mirrors initialize_from_history's final model)."""
    df = gen._add_indicators(train_df.copy())
    feats = gen._prepare_features(df)
    labels = gen._create_labels(df)
    valid = ~(feats.isna().any(axis=1) | labels.isna())
    X, y = feats[valid], labels[valid]
    if len(X) < 100 or y.sum() < 10:
        gen.is_fitted = False
        return
    gen.model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=10,
        class_weight="balanced", n_jobs=-1, random_state=42)
    gen.model.fit(gen.scaler.fit_transform(X), y)
    gen.is_fitted = True


def precompute_signals(gen, test_df):
    """Build the [momentum, mean_reversion, volume, ml] sub-signal list per bar."""
    test_df = test_df.reset_index(drop=True)
    n = len(test_df)

    # Batched ML probabilities (one predict for the whole fold)
    proba = np.full(n, np.nan)
    if gen.is_fitted:
        feats = gen._prepare_features(test_df)
        valid = ~feats.isna().any(axis=1)
        if valid.any():
            proba[valid.values] = gen.model.predict_proba(
                gen.scaler.transform(feats[valid]))[:, 1]
    ml_thr = gen.params.get("ml_threshold", 0.55)

    per_bar = []
    for i in range(n):
        row = test_df.iloc[i]
        mom = gen._check_momentum_signal(row)
        mr = gen._check_mean_reversion_signal(row)
        vol = gen._check_volume_signal(test_df.iloc[i - 1:i + 1]) if i >= 1 else None
        ml = None
        bp = proba[i]
        if not np.isnan(bp) and bp > ml_thr:   # replicate _check_ml_signal (long-only)
            ml = {"type": "ml", "direction": 1,
                  "strength": min((bp - 0.5) * 2, 1.0), "confidence": float(bp)}
        per_bar.append([mom, mr, vol, ml])
    return test_df, per_bar


def simulate(gen, test_df, per_bar, min_vote, conf_thr):
    """One position at a time, ATR stops + costs — identical to live execution."""
    gen.min_vote = min_vote
    n = len(test_df)
    trades = []
    i = 30
    while i < n - 1:
        combined = gen._combine_signals(per_bar[i])
        if not combined or combined["confidence"] < conf_thr:
            i += 1
            continue
        d = combined["direction"]
        row = test_df.iloc[i]
        px = float(row["close"])
        atr = float(row.get("atr", px * 0.02)) or px * 0.02
        if d == 1:
            stop, tgt = px - atr * ATR_STOP_MULT, px + atr * ATR_TP_MULT
            entry = px * (1 + COMMISSION + SLIPPAGE)
        else:
            stop, tgt = px + atr * ATR_STOP_MULT, px - atr * ATR_TP_MULT
            entry = px * (1 - COMMISSION - SLIPPAGE)

        exit_px = None
        j = i + 1
        lim = min(i + MAX_HOLD_BARS + 1, n)
        while j < lim:
            b = test_df.iloc[j]
            lo, hi = float(b["low"]), float(b["high"])
            if d == 1:
                if lo <= stop: exit_px = stop; break
                if hi >= tgt:  exit_px = tgt;  break
            else:
                if hi >= stop: exit_px = stop; break
                if lo <= tgt:  exit_px = tgt;  break
            j += 1
        if exit_px is None:
            j = min(j, n - 1); exit_px = float(test_df.iloc[j]["close"])
        exit_net = exit_px * (1 - COMMISSION - SLIPPAGE) if d == 1 else exit_px * (1 + COMMISSION + SLIPPAGE)
        trades.append((exit_net - entry) / entry * d)
        i = j + 1
    return trades


def stats(returns):
    if not returns:
        return dict(n=0, wr=0, aw=0, al=0, net=0, pf=0)
    r = np.array(returns)
    wins, losses = r[r > 0], r[r <= 0]
    net = float(np.prod(1 + r) - 1)
    aw = float(wins.mean()) if len(wins) else 0.0
    al = float(abs(losses.mean())) if len(losses) else 0.0
    pf = (aw * len(wins)) / (al * len(losses)) if len(losses) and al > 0 else float("inf") if len(wins) else 0.0
    return dict(n=len(r), wr=len(wins) / len(r), aw=aw, al=al, net=net, pf=pf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--splits", type=int, default=4)
    args = ap.parse_args()

    print("=" * 64)
    print(f"LINK TUNING — {args.days}d, {args.splits} folds, ATR stops, live decision path")
    print("=" * 64)
    df = fetch(SYMBOL, TIMEFRAME, args.days)
    if len(df) < 2000:
        print("not enough data"); return

    tscv = TimeSeriesSplit(n_splits=args.splits,
                           test_size=max(200, len(df) // (args.splits + 2)))

    # config_key -> list of per-trade returns aggregated across folds
    agg = {(mv, ct): [] for mv in MIN_VOTES for ct in CONF_THRS}
    fold_wr = {(mv, ct): [] for mv in MIN_VOTES for ct in CONF_THRS}

    gen = OptimizedSignalGenerator(SYMBOL, TIMEFRAME)
    for fold, (tr, te) in enumerate(tscv.split(df), 1):
        t0 = time.time()
        train_fold(gen, df.iloc[tr].copy())
        if not gen.is_fitted:
            print(f"  fold {fold}: model not fitted, skip"); continue
        test_df = gen._add_indicators(df.iloc[te].copy())
        test_df, per_bar = precompute_signals(gen, test_df)
        for (mv, ct) in agg:
            rets = simulate(gen, test_df, per_bar, mv, ct)
            agg[(mv, ct)].extend(rets)
            fold_wr[(mv, ct)].append(stats(rets)["wr"] if rets else 0.0)
        print(f"  fold {fold} done in {time.time()-t0:.0f}s (test={len(test_df):,})")

    print("\n" + "=" * 64)
    print(f"{'min_vote':>8} {'conf':>5} | {'trades':>6} {'WR':>6} {'avgW':>7} {'avgL':>7} {'net':>8} {'PF':>5}")
    print("-" * 64)
    rows = []
    for (mv, ct), rets in agg.items():
        s = stats(rets)
        rows.append((mv, ct, s))
    # sort by min_vote then conf for readability
    for mv, ct, s in sorted(rows, key=lambda x: (-x[0], -x[1])):
        pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
        print(f"{mv:>8.2f} {ct:>5.2f} | {s['n']:>6} {s['wr']:>6.1%} "
              f"{s['aw']:>6.2%} {s['al']:>6.2%} {s['net']:>+7.1%} {pf:>5}")

    # Recommendation: most trades with WR>=52% and PF>1.3 and n>=20
    cands = [(mv, ct, s) for mv, ct, s in rows
             if s["wr"] >= 0.52 and s["pf"] > 1.3 and s["n"] >= 20]
    print("\n" + "=" * 64)
    if cands:
        best = max(cands, key=lambda x: x[2]["n"])
        mv, ct, s = best
        print(f"RECOMENDADO: min_vote={mv}, confidence_threshold={ct}")
        print(f"  -> {s['n']} trades, WR {s['wr']:.1%}, net {s['net']:+.1%}, PF {s['pf']:.2f}")
    else:
        print("Nenhuma config atinge WR>=52% & PF>1.3 & n>=20.")
        print("Melhor por frequencia com net positivo:")
        pos = [(mv, ct, s) for mv, ct, s in rows if s["net"] > 0 and s["n"] >= 10]
        if pos:
            b = max(pos, key=lambda x: x[2]["n"])
            print(f"  min_vote={b[0]}, conf={b[1]}: {b[2]['n']} trades, "
                  f"WR {b[2]['wr']:.1%}, net {b[2]['net']:+.1%}, PF {b[2]['pf']:.2f}")
        else:
            print("  nenhuma config positiva com amostra minima.")


if __name__ == "__main__":
    main()
