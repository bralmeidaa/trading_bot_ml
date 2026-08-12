# -*- coding: utf-8 -*-
"""
Research harness — compare prediction targets (path A) and strategy types
(path B) on LINK/USDT, all measured the same way: walk-forward, out-of-sample,
ATR stops, real costs, long-only, one position at a time.

Path A (ML): same features, different TRAINING TARGET.
  A0  next-bar > 0.3%        (current baseline — known noise)
  A1  12-bar direction > 0
  A2  triple-barrier         (does +2.5*ATR hit before -1.5*ATR in 24 bars?)
                             -> label = the actual trade outcome we execute
  A3  12-bar move > 0.5%

Path B (rules, no ML):
  B1  trend breakout (EMA8>EMA21 & close>max20) + ATR exit
  B2  Donchian breakout (close>max20) + ATR exit
  B3  RSI<30 mean-reversion + ATR exit

Run:  python research_strategies.py --days 365 --splits 4
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

import _bootstrap  # noqa: F401  (adds project root to sys.path)
from production_trading_system import OptimizedSignalGenerator

SYMBOL, TIMEFRAME = "LINK/USDT", "5m"
COMMISSION, SLIPPAGE = 0.001, 0.0005
ATR_STOP, ATR_TP, MAX_BARS = 1.5, 2.5, 24
ML_PROB_THR = 0.55


# ----------------------------------------------------------------------------
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


# ---- label builders --------------------------------------------------------
def label_next_bar(df):
    fr = df["close"].shift(-1) / df["close"] - 1
    return np.where(fr > 0.003, 1, np.where(fr.isna(), np.nan, 0))

def label_nbar_dir(df, n=12):
    fr = df["close"].shift(-n) / df["close"] - 1
    return np.where(fr > 0, 1, np.where(fr.isna(), np.nan, 0))

def label_nbar_move(df, n=12, thr=0.005):
    fr = df["close"].shift(-n) / df["close"] - 1
    return np.where(fr > thr, 1, np.where(fr.isna(), np.nan, 0))

def label_triple_barrier(df, tp=ATR_TP, sl=ATR_STOP, max_bars=MAX_BARS):
    close = df["close"].values; high = df["high"].values; low = df["low"].values
    atr = df["atr"].values; n = len(df)
    y = np.full(n, np.nan)
    for t in range(n):
        a = atr[t]
        if np.isnan(a) or a <= 0:
            continue
        entry = close[t]; tpx = entry + tp * a; slx = entry - sl * a
        lab = 0; end = min(t + max_bars + 1, n)
        for j in range(t + 1, end):
            if low[j] <= slx:
                lab = 0; break
            if high[j] >= tpx:
                lab = 1; break
        y[t] = lab
    return y


# ---- execution sim (long-only) --------------------------------------------
def simulate(df, entries, trailing=False):
    """entries: boolean array. One position at a time, ATR stop/tp, costs."""
    df = df.reset_index(drop=True)
    n = len(df)
    close = df["close"].values; high = df["high"].values
    low = df["low"].values; atr = df["atr"].values
    rets = []
    i = 30
    while i < n - 1:
        if not entries[i]:
            i += 1; continue
        a = atr[i]
        if np.isnan(a) or a <= 0:
            i += 1; continue
        px = close[i]
        entry = px * (1 + COMMISSION + SLIPPAGE)
        stop = px - ATR_STOP * a
        tgt = px + ATR_TP * a
        exit_px = None; j = i + 1
        lim = min(i + MAX_BARS + 1, n)
        peak = px
        while j < lim:
            lo, hi = low[j], high[j]
            if trailing:                       # trail stop up as price rises
                peak = max(peak, hi)
                stop = max(stop, peak - ATR_STOP * a)
            if lo <= stop:
                exit_px = stop; break
            if hi >= tgt:
                exit_px = tgt; break
            j += 1
        if exit_px is None:
            j = min(j, n - 1); exit_px = close[j]
        exit_net = exit_px * (1 - COMMISSION - SLIPPAGE)
        rets.append((exit_net - entry) / entry)
        i = j + 1
    return rets


def stats(rets):
    if not rets:
        return dict(n=0, wr=0, aw=0, al=0, net=0, pf=0)
    r = np.array(rets)
    w, l = r[r > 0], r[r <= 0]
    aw = float(w.mean()) if len(w) else 0.0
    al = float(abs(l.mean())) if len(l) else 0.0
    pf = (aw * len(w)) / (al * len(l)) if len(l) and al > 0 else (float("inf") if len(w) else 0.0)
    return dict(n=len(r), wr=len(w) / len(r), aw=aw, al=al,
                net=float(np.prod(1 + r) - 1), pf=pf)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--splits", type=int, default=4)
    ap.add_argument("--symbol", type=str, default=SYMBOL)
    ap.add_argument("--timeframe", type=str, default=TIMEFRAME)
    args = ap.parse_args()

    print("=" * 70)
    print(f"RESEARCH — {args.symbol} {args.timeframe}, {args.days}d, {args.splits} folds, "
          f"ATR {ATR_STOP}/{ATR_TP}, long-only")
    print("=" * 70)
    raw = fetch(args.symbol, args.timeframe, args.days)
    gen = OptimizedSignalGenerator(args.symbol, args.timeframe)
    df = gen._add_indicators(raw.copy()).reset_index(drop=True)

    ml_targets = {
        "A0 next-bar>0.3%":  label_next_bar,
        "A1 12bar dir>0":    label_nbar_dir,
        "A2 triple-barrier": label_triple_barrier,
        "A3 12bar move>0.5%":label_nbar_move,
    }

    tscv = TimeSeriesSplit(n_splits=args.splits,
                           test_size=max(200, len(df) // (args.splits + 2)))
    folds = list(tscv.split(df))

    # accumulators
    ml_rets = {k: [] for k in ml_targets}
    ml_lift = {k: [] for k in ml_targets}
    rule_rets = {"B1 trend+trail": [], "B2 donchian": [], "B3 rsi<30": []}

    feat_cols = ["sma_20", "ema_8", "ema_21", "rsi", "bb_position",
                 "atr", "volume_ratio", "momentum_5", "momentum_10"]

    for fi, (tr, te) in enumerate(folds, 1):
        t0 = time.time()
        train = df.iloc[tr]; test = df.iloc[te]
        Xtr_raw = train[feat_cols].ffill().fillna(0)
        Xte_raw = test[feat_cols].ffill().fillna(0)

        # ---- ML variants ----
        for name, fn in ml_targets.items():
            ytr = fn(train); yte = fn(test)
            ytr = pd.Series(ytr, index=train.index)
            yte = pd.Series(yte, index=test.index)
            mtr = ~ytr.isna(); mte = ~yte.isna()
            if mtr.sum() < 200 or ytr[mtr].sum() < 20:
                continue
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr_raw[mtr.values])
            clf = RandomForestClassifier(n_estimators=150, max_depth=6,
                                         min_samples_leaf=10, class_weight="balanced",
                                         n_jobs=-1, random_state=42)
            clf.fit(Xtr, ytr[mtr].values)
            # lift on test
            Xte = sc.transform(Xte_raw[mte.values])
            acc = accuracy_score(yte[mte].values, clf.predict(Xte))
            base = max(yte[mte].mean(), 1 - yte[mte].mean())
            ml_lift[name].append(acc - base)
            # economic: enter when proba>thr
            proba_all = np.zeros(len(test))
            proba_all[mte.values] = clf.predict_proba(Xte)[:, 1]
            entries = proba_all > ML_PROB_THR
            ml_rets[name].extend(simulate(test, entries))

        # ---- rule variants ----
        ema8 = test["ema_8"].values; ema21 = test["ema_21"].values
        close = test["close"].values
        max20 = test["close"].rolling(20).max().shift(1).values
        rsi = test["rsi"].values
        b1 = (ema8 > ema21) & (close > np.nan_to_num(max20, nan=np.inf))
        b2 = close > np.nan_to_num(max20, nan=np.inf)
        b3 = rsi < 30
        rule_rets["B1 trend+trail"].extend(simulate(test, b1, trailing=True))
        rule_rets["B2 donchian"].extend(simulate(test, b2))
        rule_rets["B3 rsi<30"].extend(simulate(test, b3))

        print(f"  fold {fi} done in {time.time()-t0:.0f}s")

    # ---- report ----
    print("\n" + "=" * 70)
    print(f"{'variant':<20} {'lift':>6} | {'trades':>6} {'WR':>6} {'avgW':>7} {'avgL':>7} {'net':>8} {'PF':>5}")
    print("-" * 70)

    def row(name, rets, lift=None):
        s = stats(rets)
        lf = f"{np.mean(lift):+.3f}" if lift else "   -- "
        pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
        print(f"{name:<20} {lf:>6} | {s['n']:>6} {s['wr']:>6.1%} "
              f"{s['aw']:>6.2%} {s['al']:>6.2%} {s['net']:>+7.1%} {pf:>5}")
        return s

    print("  [A] PREDICTION TARGETS (ML)")
    a_results = {}
    for name in ml_targets:
        a_results[name] = row(name, ml_rets[name], ml_lift[name])
    print("  [B] STRATEGY TYPES (rules)")
    b_results = {}
    for name in rule_rets:
        b_results[name] = row(name, rule_rets[name])

    # ---- verdict ----
    print("\n" + "=" * 70)
    all_res = {**a_results, **b_results}
    winners = [(k, s) for k, s in all_res.items()
               if s["net"] > 0 and s["wr"] >= 0.48 and s["pf"] > 1.15 and s["n"] >= 20]
    if winners:
        best = max(winners, key=lambda x: x[1]["net"])
        print(f"MELHOR: {best[0]} -> {best[1]['n']} trades, WR {best[1]['wr']:.1%}, "
              f"net {best[1]['net']:+.1%}, PF {best[1]['pf']:.2f}")
        print("Candidatos positivos:")
        for k, s in sorted(winners, key=lambda x: -x[1]["net"]):
            print(f"  {k}: net {s['net']:+.1%}, WR {s['wr']:.1%}, PF {s['pf']:.2f}, n={s['n']}")
    else:
        print("Nenhuma variante atinge net>0 & WR>=48% & PF>1.15 & n>=20.")
        print("Leitura: nem alvo melhor nem tipo de estrategia destrava edge com")
        print("estas features/par. Proximo: enriquecer features OU trocar par/timeframe.")
    # ML lift summary
    print("\nLift do ML (accuracy - baseline; >0 = modelo agrega informacao):")
    for name in ml_targets:
        if ml_lift[name]:
            print(f"  {name}: {np.mean(ml_lift[name]):+.3f}")


if __name__ == "__main__":
    main()
