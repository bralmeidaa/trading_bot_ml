# -*- coding: utf-8 -*-
"""
Feature research — does giving the ML actual candle SHAPES / sequences (and
external data like funding rate) raise predictive lift above ~0?

Answers the question: "can the model find trend/candle patterns from history
if we actually encode them?" — by adding those features and measuring lift
out-of-sample (walk-forward).

Feature sets:
  F0  baseline 9 TA indicators (current production set)
  F1  F0 + candle-shape/sequence features (body, wicks, close position,
      consecutive direction, last-N returns, distance from MAs, volatility,
      simple patterns: doji/hammer/engulfing)
  F2  F1 + funding rate (Binance USD-M perp, forward-filled, point-in-time)

Targets: A1 (12-bar direction) and A2 (triple-barrier = the trade outcome).
Key metric: LIFT = out-of-sample accuracy − majority-class baseline.
>0 means the features add real predictive information.

Run:  python research_features.py --days 365 --splits 4
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from production_trading_system import OptimizedSignalGenerator

SYMBOL, TIMEFRAME = "LINK/USDT", "5m"
ATR_STOP, ATR_TP, MAX_BARS = 1.5, 2.5, 24


def fetch_ohlcv(symbol, timeframe, days):
    ex = ccxt.binance({"enableRateLimit": True, "rateLimit": 1200})
    tf_min = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}.get(timeframe, 5)
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    out = []
    print(f"  ohlcv {symbol} {timeframe} {days}d...", end="", flush=True)
    while True:
        b = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not b:
            break
        out.extend(b); since = b[-1][0] + tf_min * 60 * 1000
        if len(b) < 1000:
            break
        time.sleep(0.25)
    df = pd.DataFrame(out, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    print(f" {len(df):,}")
    return df


def fetch_funding(symbol, days):
    """Funding rate history from Binance USD-M perp (every 8h). Returns df[ts,funding] or None."""
    try:
        fut = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        rows = []
        print("  funding...", end="", flush=True)
        while True:
            batch = fut.fetch_funding_rate_history(symbol, since=since, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            since = batch[-1]["timestamp"] + 1
            if len(batch) < 1000:
                break
            time.sleep(0.25)
        if not rows:
            print(" none"); return None
        fr = pd.DataFrame([{"timestamp": r["timestamp"], "funding": r["fundingRate"]} for r in rows])
        fr = fr.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        print(f" {len(fr)} points")
        return fr
    except Exception as e:
        print(f" failed ({e})")
        return None


# ---- feature builders ------------------------------------------------------
BASE = ["sma_20", "ema_8", "ema_21", "rsi", "bb_position",
        "atr", "volume_ratio", "momentum_5", "momentum_10"]


def add_candle_features(df):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0, np.nan)
    df["f_body"] = (c - o) / o
    df["f_upwick"] = (h - np.maximum(o, c)) / o
    df["f_lowick"] = (np.minimum(o, c) - l) / o
    df["f_range"] = (h - l) / o
    df["f_closepos"] = (c - l) / rng           # where in the bar it closed (0=low,1=high)
    df["f_bull"] = (c > o).astype(float)
    # last-N returns (recent sequence/shape)
    for k in (1, 2, 3, 5, 8):
        df[f"f_ret{k}"] = c.pct_change(k)
    # consecutive bull/bear count over last 5
    bull = (c > o).astype(int)
    df["f_consec"] = bull.rolling(5).sum()
    # distance from moving averages (trend position)
    df["f_dist_sma20"] = (c - df["sma_20"]) / df["sma_20"]
    df["f_dist_ema8"] = (c - df["ema_8"]) / df["ema_8"]
    df["f_dist_ema21"] = (c - df["ema_21"]) / df["ema_21"]
    df["f_ema_spread"] = (df["ema_8"] - df["ema_21"]) / df["ema_21"]
    # rolling volatility of returns
    df["f_vol10"] = c.pct_change().rolling(10).std()
    df["f_vol30"] = c.pct_change().rolling(30).std()
    # simple patterns
    body = (c - o).abs()
    df["f_doji"] = (body <= 0.1 * (h - l)).astype(float)
    df["f_hammer"] = ((df["f_lowick"] > 2 * body / o) & (df["f_upwick"] < body / o)).astype(float)
    prev_o, prev_c = o.shift(1), c.shift(1)
    df["f_bull_engulf"] = (((c > o) & (prev_c < prev_o) &
                            (c >= prev_o) & (o <= prev_c))).astype(float)
    return df


CANDLE = ["f_body", "f_upwick", "f_lowick", "f_range", "f_closepos", "f_bull",
          "f_ret1", "f_ret2", "f_ret3", "f_ret5", "f_ret8", "f_consec",
          "f_dist_sma20", "f_dist_ema8", "f_dist_ema21", "f_ema_spread",
          "f_vol10", "f_vol30", "f_doji", "f_hammer", "f_bull_engulf"]


# ---- targets ---------------------------------------------------------------
def label_nbar_dir(df, n=12):
    fr = df["close"].shift(-n) / df["close"] - 1
    return np.where(fr > 0, 1.0, np.where(fr.isna(), np.nan, 0.0))

def label_triple_barrier(df, tp=ATR_TP, sl=ATR_STOP, max_bars=MAX_BARS):
    close = df["close"].values; high = df["high"].values; low = df["low"].values
    atr = df["atr"].values; n = len(df); y = np.full(n, np.nan)
    for t in range(n):
        a = atr[t]
        if np.isnan(a) or a <= 0:
            continue
        entry = close[t]; tpx = entry + tp * a; slx = entry - sl * a
        lab = 0; end = min(t + max_bars + 1, n)
        for j in range(t + 1, end):
            if low[j] <= slx: lab = 0; break
            if high[j] >= tpx: lab = 1; break
        y[t] = lab
    return y

TARGETS = {"A1 12bar-dir": label_nbar_dir, "A2 triple-barrier": label_triple_barrier}


def measure(df, feat_cols, target_fn, folds):
    """Return mean out-of-sample lift across folds for one (features,target)."""
    lifts = []
    for tr, te in folds:
        train, test = df.iloc[tr], df.iloc[te]
        ytr = pd.Series(target_fn(train), index=train.index)
        yte = pd.Series(target_fn(test), index=test.index)
        Xtr_raw = train[feat_cols].ffill().fillna(0)
        Xte_raw = test[feat_cols].ffill().fillna(0)
        mtr = ~ytr.isna(); mte = ~yte.isna()
        if mtr.sum() < 200 or ytr[mtr].sum() < 20 or mte.sum() < 50:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr_raw[mtr.values])
        clf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=10,
                                     class_weight="balanced", n_jobs=-1, random_state=42)
        clf.fit(Xtr, ytr[mtr].values)
        Xte = sc.transform(Xte_raw[mte.values])
        acc = accuracy_score(yte[mte].values, clf.predict(Xte))
        base = max(yte[mte].mean(), 1 - yte[mte].mean())
        lifts.append(acc - base)
    return float(np.mean(lifts)) if lifts else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--splits", type=int, default=4)
    args = ap.parse_args()

    print("=" * 64)
    print(f"FEATURE RESEARCH — {SYMBOL} {TIMEFRAME}, {args.days}d, {args.splits} folds")
    print("=" * 64)
    raw = fetch_ohlcv(SYMBOL, TIMEFRAME, args.days)
    gen = OptimizedSignalGenerator(SYMBOL, TIMEFRAME)
    df = gen._add_indicators(raw.copy())
    df = add_candle_features(df).reset_index(drop=True)

    # funding rate (optional)
    fr = fetch_funding(SYMBOL, args.days)
    has_funding = False
    if fr is not None and len(fr) > 5:
        df = pd.merge_asof(df.sort_values("timestamp"), fr.sort_values("timestamp"),
                           on="timestamp", direction="backward")
        df["f_funding"] = df["funding"].ffill().fillna(0)
        has_funding = True

    feature_sets = {
        "F0 baseline TA": BASE,
        "F1 +candle/seq": BASE + CANDLE,
    }
    if has_funding:
        feature_sets["F2 +funding"] = BASE + CANDLE + ["f_funding"]

    tscv = TimeSeriesSplit(n_splits=args.splits,
                           test_size=max(200, len(df) // (args.splits + 2)))
    folds = list(tscv.split(df))

    print("\n" + "=" * 64)
    print("LIFT (out-of-sample accuracy - baseline). >0 = features add signal.")
    print("=" * 64)
    print(f"{'feature set':<18}" + "".join(f"{t:>20}" for t in TARGETS))
    print("-" * 64)
    best = (None, None, -1)
    for fname, cols in feature_sets.items():
        cells = []
        for tname, tfn in TARGETS.items():
            t0 = time.time()
            lift = measure(df, cols, tfn, folds)
            cells.append(lift)
            if not np.isnan(lift) and lift > best[2]:
                best = (fname, tname, lift)
        print(f"{fname:<18}" + "".join(f"{c:>+20.3f}" for c in cells))

    print("\n" + "=" * 64)
    print(f"Melhor lift: {best[0]} / {best[1]} = {best[2]:+.3f}")
    if best[2] <= 0.01:
        print("VEREDITO: mesmo com formato de candle/sequencia" +
              (" + funding" if has_funding else "") + ", o lift continua ~0/negativo.")
        print("Os padroes nao preveem o futuro de forma generalizavel neste par/timeframe.")
    else:
        print(f"VEREDITO: features novas elevam o lift para {best[2]:+.3f} (>0). Vale aprofundar.")


if __name__ == "__main__":
    main()
