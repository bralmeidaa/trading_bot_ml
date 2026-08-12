# -*- coding: utf-8 -*-
"""
Intraday research harness — validates the ORB+VWAP+session strategy honestly.

Pipeline: generate setups → simulate triple-barrier outcomes → train an AI
META-FILTER (predict P(win) from context) walk-forward → trade only setups the
filter approves → report per-trade expectancy AND CONCRETE PROFIT (weekly /
monthly, in % and $ on the capital), net of cost.

Unit of analysis = SETUPS, not bars. Win rate alone is not profit; profit =
trades/period × expectancy. The report makes that explicit.

Run:
  python .claude/research/research_intraday.py --symbol BTC/USDT --timeframe 5m --days 180
  python .claude/research/research_intraday.py --sweep
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import _bootstrap  # noqa: F401

import os
import argparse
from datetime import datetime, timedelta

import ccxt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from backend.strategy.intraday_setups import SetupConfig, generate_setups, simulate_setups
from backend.data.intraday_features import CONTEXT_FEATURES

COST_PER_SIDE = 0.0005          # 0.05%/side (maker-ish on majors)
CAPITAL = 1200.0
RISK_PER_TRADE = 0.01           # 1% of capital risked per trade
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
TF_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}


def fetch(symbol, timeframe, days):
    ex = ccxt.binance({"enableRateLimit": True, "rateLimit": 1200})
    tf_min = TF_MIN.get(timeframe, 5)
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    out = []
    while True:
        b = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not b:
            break
        out.extend(b); since = b[-1][0] + tf_min * 60 * 1000
        if len(b) < 1000:
            break
    df = pd.DataFrame(out, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def expectancy_report(trades, label, days, tf, capital=CAPITAL, risk=RISK_PER_TRADE):
    """Translate a list of trades into CONCRETE profit projections."""
    if not trades:
        return None
    r = np.array([t["net_r"] for t in trades])
    wins = (r > 0).sum()
    n = len(r)
    win_rate = wins / n
    exp_r = r.mean()                       # expectancy in R per trade
    # %-return per trade = R-multiple × risk_per_trade (sizing by fixed fractional risk)
    ret_per_trade = exp_r * risk
    trades_per_day = n / max(days, 1)
    weekly_ret = trades_per_day * 7 * ret_per_trade
    monthly_ret = trades_per_day * 30 * ret_per_trade
    # equity curve in R, then in $ (each trade risks `risk*capital`)
    pnl_dollars = np.cumsum(r * risk * capital)
    equity = capital + pnl_dollars
    peak = np.maximum.accumulate(equity)
    maxdd = float(((equity - peak) / peak).min()) if len(equity) else 0.0
    return {
        "label": label, "n": n, "win_rate": win_rate, "exp_r": exp_r,
        "trades_per_day": trades_per_day,
        "weekly_ret": weekly_ret, "monthly_ret": monthly_ret,
        "weekly_usd": weekly_ret * capital, "monthly_usd": monthly_ret * capital,
        "maxdd": maxdd, "total_net_usd": float(pnl_dollars[-1]) if len(pnl_dollars) else 0.0,
    }


def fold_consistency(trades, n_folds=4):
    """Chronological split: per-fold expectancy (R) + how many folds positive.
    Tests whether the raw setup edge holds across sub-periods (not just overall)."""
    if len(trades) < n_folds * 5:
        return {"per_fold": [], "pos_folds": 0, "n_folds": 0}
    r = np.array([t["net_r"] for t in trades])
    folds = np.array_split(r, n_folds)
    exps = [float(f.mean()) for f in folds if len(f)]
    return {"per_fold": exps, "pos_folds": sum(1 for e in exps if e > 0),
            "n_folds": len(exps)}


def meta_filter_walk_forward(df, trades, n_splits=4, prob_thr=0.55):
    """
    Train an AI meta-filter walk-forward: predict P(win) from setup context.
    Returns the subset of OOS trades the filter approved (prob > thr).
    """
    if len(trades) < 60:
        return trades, "too few setups for meta-filter (using raw)"
    X = pd.DataFrame([t["context"] for t in trades]).reindex(columns=CONTEXT_FEATURES).fillna(0.0)
    y = np.array([1 if t["net_r"] > 0 else 0 for t in trades])
    idx = np.arange(len(trades))
    approved = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr, te in tscv.split(idx):
        if y[tr].sum() < 5 or (len(tr) - y[tr].sum()) < 5:
            continue
        sc = StandardScaler()
        clf = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=10,
                                     class_weight="balanced", n_jobs=-1, random_state=42)
        clf.fit(sc.fit_transform(X.iloc[tr]), y[tr])
        proba = clf.predict_proba(sc.transform(X.iloc[te]))[:, 1]
        for j, p in zip(te, proba):
            if p > prob_thr:
                approved.append(trades[j])
    return approved, f"meta-filter approved {len(approved)}/{len(trades)} OOS setups"


def run_one(symbol, timeframe, days, cfg, n_splits=4):
    raw = fetch(symbol, timeframe, days)
    if len(raw) < 1000:
        print(f"  {symbol} {timeframe}: insufficient data ({len(raw)})")
        return None
    bar_min = TF_MIN.get(timeframe, 5)
    cfg.bar_minutes = bar_min
    setups = generate_setups(raw, symbol, cfg)
    if len(setups) < 30:
        print(f"  {symbol} {timeframe}: only {len(setups)} setups — skip")
        return None
    trades = simulate_setups(raw, setups, cost_per_side=COST_PER_SIDE)
    raw_rep = expectancy_report(trades, "setups crus", days, timeframe)
    fc = fold_consistency(trades)
    approved, note = meta_filter_walk_forward(trades, trades, n_splits)
    flt_rep = expectancy_report(approved, "com meta-filtro IA", days, timeframe)
    return {"raw": raw_rep, "filtered": flt_rep, "note": note, "fold": fc,
            "n_setups": len(setups), "symbol": symbol, "tf": timeframe}


def _print_rep(rep):
    if not rep:
        print("    (sem trades)"); return
    print(f"    {rep['label']:<20} n={rep['n']:<4} WR={rep['win_rate']:.0%} "
          f"exp={rep['exp_r']:+.2f}R  trades/dia={rep['trades_per_day']:.2f}")
    print(f"      → LUCRO: semanal {rep['weekly_ret']:+.2%} (${rep['weekly_usd']:+.0f})  "
          f"mensal {rep['monthly_ret']:+.2%} (${rep['monthly_usd']:+.0f})  maxDD {rep['maxdd']:+.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="5m")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--cost", type=float, default=None, help="override cost per side")
    ap.add_argument("--stop-mult", type=float, default=1.0)
    ap.add_argument("--rr", type=float, default=2.0)
    args = ap.parse_args()

    global COST_PER_SIDE
    if args.cost is not None:
        COST_PER_SIDE = args.cost

    cfg = SetupConfig(rr=args.rr, stop_atr_mult=args.stop_mult, require_trend=True,
                      require_vwap=True, min_vol_ratio=1.0)

    combos = ([("BTC/USDT", tf) for tf in ("5m", "15m", "1h")] +
              [("ETH/USDT", tf) for tf in ("5m", "15m", "1h")]) if args.sweep \
             else [(args.symbol, args.timeframe)]

    print("=" * 74)
    print(f"INTRADAY RESEARCH — ORB+VWAP+sessão | capital ${CAPITAL:.0f}, "
          f"risco {RISK_PER_TRADE:.0%}/trade, custo {COST_PER_SIDE*100:.2f}%/lado")
    print("=" * 74)

    results, lines = [], []
    for sym, tf in combos:
        print(f"\n{sym} {tf} ({args.days}d):")
        res = run_one(sym, tf, args.days, SetupConfig(**vars(cfg)) if False else cfg)
        if not res:
            continue
        fc = res["fold"]
        print(f"  {res['n_setups']} setups | {res['note']}")
        if fc["n_folds"]:
            print(f"  consistência (R por fold): {[round(e,2) for e in fc['per_fold']]} "
                  f"→ {fc['pos_folds']}/{fc['n_folds']} folds positivos")
        _print_rep(res["raw"])
        _print_rep(res["filtered"])
        results.append(res)

    # verdict: raw edge must hold across MOST folds AND be net-positive monthly
    print("\n" + "=" * 74)
    go = []
    for res in results:
        raw, fc = res["raw"], res["fold"]
        consistent = fc["n_folds"] >= 3 and fc["pos_folds"] >= fc["n_folds"] - 1
        if raw and raw["monthly_ret"] > 0 and consistent and raw["n"] >= 30:
            go.append((res["symbol"], res["tf"], raw, fc))
    if go:
        print("CANDIDATOS (edge cru consistente nos folds + lucro mensal positivo):")
        for sym, tf, raw, fc in sorted(go, key=lambda x: -x[2]["monthly_ret"]):
            print(f"  {sym} {tf}: mensal {raw['monthly_ret']:+.2%} (${raw['monthly_usd']:+.0f}), "
                  f"WR {raw['win_rate']:.0%}, exp {raw['exp_r']:+.2f}R, "
                  f"{fc['pos_folds']}/{fc['n_folds']} folds+, maxDD {raw['maxdd']:+.1%}")
        print("\nVEREDITO: há setup(s) com edge consistente — validar mais (mais histórico, "
              "paper Fase 7). Atenção: extrapolação de lucro assume estacionariedade.")
    else:
        print("VEREDITO: nenhum setup com edge CONSISTENTE nos folds + lucro positivo.")
        print("Sinal cru pode aparecer num período, mas não se sustenta — sem edge robusto.")

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "intraday_report.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Intraday ORB+VWAP+sessão — relatório\n\nCapital ${CAPITAL:.0f}, "
                 f"risco {RISK_PER_TRADE:.0%}/trade, custo {COST_PER_SIDE*100:.2f}%/lado\n\n")
        fh.write("| symbol | tf | n | WR | exp(R) | semanal% | mensal% | mensal$ | maxDD |\n")
        fh.write("|--------|----|---|----|--------|----------|---------|---------|-------|\n")
        for res in results:
            f = res["filtered"] or res["raw"]
            if f:
                fh.write(f"| {res['symbol']} | {res['tf']} | {f['n']} | {f['win_rate']:.0%} | "
                         f"{f['exp_r']:+.2f} | {f['weekly_ret']:+.2%} | {f['monthly_ret']:+.2%} | "
                         f"${f['monthly_usd']:+.0f} | {f['maxdd']:+.1%} |\n")
    print(f"\nRelatório salvo em {os.path.join(REPORT_DIR, 'intraday_report.md')}")


if __name__ == "__main__":
    main()
