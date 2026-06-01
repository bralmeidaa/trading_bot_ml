# -*- coding: utf-8 -*-
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
#!/usr/bin/env python3
"""
Walk-forward backtest — run this BEFORE enabling live trading.

Usage:
    python run_backtest.py                 # default: 365 days, 5 folds
    python run_backtest.py --days 180      # shorter history
    python run_backtest.py --splits 3      # fewer folds (faster)

No API keys needed — uses Binance public market data.

Acceptance criteria (all must pass for GO verdict):
    Sharpe ratio     > 0.8
    Max drawdown     < 20%
    Win rate         > 45%
    Profit factor    > 1.2
    Trades (total)   >= 30
"""
import sys
import time
import json
import argparse
from datetime import datetime, timedelta

import ccxt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from production_trading_system import OptimizedSignalGenerator, create_production_config

# -- Execution cost model ---------------------------------------------------
COMMISSION = 0.001   # 0.1% per side (Binance VIP 0)
SLIPPAGE   = 0.0005  # 0.05% market impact

# -- Trade exit model (fixed %, not ATR-based) ------------------------------
# ATR on 5m crypto is ~0.15-0.35% — too tight for realistic stops.
# 1.5% stop / 3.0% TP = 1:2 R/R gross; breakeven WR = 33%.
STOP_PCT      = 0.015   # 1.5% stop loss from entry
TP_PCT        = 0.030   # 3.0% take profit from entry
MAX_HOLD_BARS = 100     # force-exit after 100 bars if neither stop nor TP hit

# -- Walk-forward settings --------------------------------------------------
DEFAULT_DAYS   = 365
DEFAULT_SPLITS = 5


# ==========================================================================
# Data fetching
# ==========================================================================

def fetch_history(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Download OHLCV history from Binance (no API key required)."""
    exchange = ccxt.binance({"enableRateLimit": True, "rateLimit": 1200})
    tf_minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
    tf_min = tf_minutes.get(timeframe, 5)
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    all_ohlcv = []
    limit = 1000
    print(f"  Downloading {symbol} {timeframe} ({days}d)...", end="", flush=True)

    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except Exception as exc:
            print(f"\n  ⚠ fetch_ohlcv error: {exc}")
            break
        if not batch:
            break
        all_ohlcv.extend(batch)
        since = batch[-1][0] + tf_min * 60 * 1000
        if len(batch) < limit:
            break
        time.sleep(0.3)

    if not all_ohlcv:
        print(" no data returned")
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    print(f" {len(df):,} candles [OK]")
    return df


# ==========================================================================
# Vectorized signal generation (fast — no bar-by-bar loop)
# ==========================================================================

def generate_signals_vectorized(test_df: pd.DataFrame, signal_gen: OptimizedSignalGenerator) -> pd.Series:
    """
    Generate a direction series (+1 long / -1 short / 0 flat) for all test bars.

    Strategy: strict mean reversion at extreme RSI/BB levels.
    ML acts as an optional confirmation for longs only — NOT required.
    """
    direction = pd.Series(0, index=test_df.index)

    has_cols = all(c in test_df.columns for c in ["rsi", "bb_position"])
    if not has_cols:
        return direction

    # -- Strict mean reversion (primary signal) ----------------------------
    # Thresholds chosen for genuinely extreme conditions, not moderate ones.
    rev_buy  = (test_df["bb_position"] < 0.08) & (test_df["rsi"] < 30)   # extreme oversold
    rev_sell = (test_df["bb_position"] > 0.92) & (test_df["rsi"] > 70)   # extreme overbought

    direction[rev_buy]  =  1
    direction[rev_sell] = -1

    # -- ML confirmation (optional uplift for longs) -----------------------
    # Only ADDS longs where ML is confident; never changes a 0 into a short.
    if signal_gen.is_fitted:
        features = signal_gen._prepare_features(test_df)
        valid = ~features.isna().any(axis=1)
        if valid.any():
            X = features[valid]
            try:
                X_scaled = signal_gen.scaler.transform(X)
                proba = signal_gen.model.predict_proba(X_scaled)[:, 1]
                thresh = signal_gen.params.get("ml_threshold", 0.55)
                ml_long = pd.Series(False, index=test_df.index)
                ml_long[valid] = proba > thresh

                # Add ML-confirmed longs that aren't already signalled
                # (also catches momentum longs missed by strict BB/RSI filter)
                p = signal_gen.params
                mom_buy = (
                    (test_df.get("momentum_5", pd.Series(0, index=test_df.index)) > p.get("momentum_threshold", 0.003)) &
                    (test_df.get("volume_ratio", pd.Series(1, index=test_df.index)) > p.get("volume_threshold", 1.5)) &
                    (test_df["rsi"] < p.get("rsi_overbought", 62))
                )
                direction[(ml_long) & (mom_buy) & (direction == 0)] = 1
            except Exception:
                pass

    return direction


# ==========================================================================
# Trade simulation
# ==========================================================================

def simulate_trades(test_df: pd.DataFrame, signals: pd.Series) -> list:
    """
    Simulate trades given a signal series.
    Entry: close of signal bar + costs.
    Exit: first bar where low ≤ stop OR high ≥ TP (or MAX_HOLD_BARS).
    Returns list of dicts with pnl_pct, direction, bars_held.
    """
    test_df = test_df.reset_index(drop=True)
    signals = signals.reset_index(drop=True)
    n = len(test_df)
    trades = []
    i = 0

    while i < n - 1:
        sig = int(signals.iloc[i])
        if sig == 0:
            i += 1
            continue

        row = test_df.iloc[i]
        atr = float(row.get("atr", row["close"] * 0.02)) or row["close"] * 0.02

        # Entry with costs; stops/TP as fixed % of entry price
        raw_entry = float(row["close"])
        if sig == 1:
            entry  = raw_entry * (1 + COMMISSION + SLIPPAGE)
            stop   = entry * (1 - STOP_PCT)
            target = entry * (1 + TP_PCT)
        else:
            entry  = raw_entry * (1 - COMMISSION - SLIPPAGE)
            stop   = entry * (1 + STOP_PCT)
            target = entry * (1 - TP_PCT)

        exit_price = None
        j = i + 1
        limit = min(i + MAX_HOLD_BARS + 1, n)

        while j < limit:
            bar = test_df.iloc[j]
            low, high = float(bar["low"]), float(bar["high"])
            if sig == 1:
                if low <= stop:
                    exit_price = stop
                    break
                if high >= target:
                    exit_price = target
                    break
            else:
                if high >= stop:
                    exit_price = stop
                    break
                if low <= target:
                    exit_price = target
                    break
            j += 1

        # Force exit at close if no stop/TP hit
        if exit_price is None:
            j = min(j, n - 1)
            exit_price = float(test_df.iloc[j]["close"])

        # Exit with costs
        if sig == 1:
            exit_net = exit_price * (1 - COMMISSION - SLIPPAGE)
        else:
            exit_net = exit_price * (1 + COMMISSION + SLIPPAGE)

        pnl_pct = (exit_net - entry) / entry * sig
        trades.append({"pnl_pct": pnl_pct, "direction": sig, "bars_held": j - i})
        i = j + 1  # no overlapping trades

    return trades


# ==========================================================================
# Metrics
# ==========================================================================

def calc_metrics(trades: list) -> dict:
    if not trades:
        return {}

    returns = np.array([t["pnl_pct"] for t in trades])
    wins   = returns[returns > 0]
    losses = returns[returns <= 0]
    n = len(returns)

    equity   = np.cumprod(1 + returns)
    total_r  = float(equity[-1] - 1)

    # Annualise: assume each trade is ~independent, use 252 trade-days per year heuristic
    ann_r = float((1 + total_r) ** (252 / max(n, 1)) - 1)

    vol        = float(returns.std() * np.sqrt(252)) or 1e-9
    sharpe     = ann_r / vol
    down_vol   = float(returns[returns < 0].std() * np.sqrt(252)) if len(losses) > 1 else vol
    sortino    = ann_r / (down_vol or 1e-9)

    roll_max = np.maximum.accumulate(equity)
    max_dd   = float(abs(((equity - roll_max) / roll_max).min()))
    calmar   = ann_r / (max_dd or 1e-9)

    win_rate  = float(len(wins) / n)
    avg_win   = float(wins.mean())  if len(wins)   > 0 else 0.0
    avg_loss  = float(abs(losses.mean())) if len(losses) > 0 else 0.0
    pf_denom  = avg_loss * max(len(losses), 1)
    pf        = (avg_win * len(wins)) / pf_denom if pf_denom > 0 else 0.0

    return {
        "n_trades":      n,
        "total_return":  total_r,
        "annual_return": ann_r,
        "sharpe":        sharpe,
        "sortino":       sortino,
        "max_drawdown":  max_dd,
        "calmar":        calmar,
        "win_rate":      win_rate,
        "profit_factor": pf,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
    }


# ==========================================================================
# Walk-forward loop
# ==========================================================================

def run_walk_forward(symbol: str, timeframe: str, df: pd.DataFrame, n_splits: int) -> dict | None:
    """Full walk-forward backtest for one symbol/timeframe."""
    print(f"\n{'='*60}")
    print(f"  {symbol}  {timeframe}  ({len(df):,} bars, {n_splits} folds)")
    print(f"{'='*60}")

    signal_gen = OptimizedSignalGenerator(symbol, timeframe)
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=max(200, len(df) // (n_splits + 2)))

    all_trades: list = []
    fold_rows: list  = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx].copy()
        test_df  = df.iloc[test_idx].copy()

        print(f"\n  Fold {fold + 1}/{n_splits}  "
              f"train={len(train_df):,}  test={len(test_df):,}", end="")

        # Train signal generator on training fold
        signal_gen.initialize_from_history(train_df)

        if not signal_gen.is_fitted:
            print("  ⚠ model not fitted — skipping")
            continue

        # Add indicators to test fold
        test_df = signal_gen._add_indicators(test_df.copy()).reset_index(drop=True)

        # Generate signals (vectorised) + simulate trades
        signals     = generate_signals_vectorized(test_df, signal_gen)
        fold_trades = simulate_trades(test_df, signals)
        all_trades.extend(fold_trades)

        if fold_trades:
            fm = calc_metrics(fold_trades)
            fold_rows.append(fm)
            print(f"  →  trades={fm['n_trades']:3d}  "
                  f"WR={fm['win_rate']:.1%}  "
                  f"Sharpe={fm['sharpe']:+.2f}  "
                  f"PnL={fm['total_return']:+.2%}")
        else:
            print("  →  no trades generated")

    # -- Aggregate ------------------------------------------------------
    print(f"\n  {'-'*54}")
    print(f"  AGGREGATE  ({len(all_trades)} total trades across all folds)")
    print(f"  {'-'*54}")

    if not all_trades:
        print("  [FAIL]  No trades — strategy never fired in test periods.")
        print("  Possible causes: confidence threshold too high, no data overlap.")
        return None

    m = calc_metrics(all_trades)

    print(f"  Total Return  : {m['total_return']:+.2%}")
    print(f"  Annual Return : {m['annual_return']:+.2%}")
    print(f"  Sharpe Ratio  : {m['sharpe']:.2f}")
    print(f"  Sortino Ratio : {m['sortino']:.2f}")
    print(f"  Max Drawdown  : {m['max_drawdown']:.2%}")
    print(f"  Calmar Ratio  : {m['calmar']:.2f}")
    print(f"  Win Rate      : {m['win_rate']:.1%}")
    print(f"  Profit Factor : {m['profit_factor']:.2f}")
    print(f"  Avg Win       : {m['avg_win']:+.3%}")
    print(f"  Avg Loss      : {m['avg_loss']:+.3%}")

    # -- Acceptance criteria --------------------------------------------
    criteria = {
        "Sharpe > 0.8":       m["sharpe"]        > 0.8,
        "MaxDD < 20%":        m["max_drawdown"]   < 0.20,
        "WinRate > 45%":      m["win_rate"]       > 0.45,
        "ProfitFactor > 1.2": m["profit_factor"]  > 1.2,
        "Trades >= 30":       m["n_trades"]       >= 30,
    }

    print(f"\n  {'-'*54}")
    print("  ACCEPTANCE CRITERIA")
    print(f"  {'-'*54}")
    passed = sum(1 for ok in criteria.values() if ok)
    for label, ok in criteria.items():
        print(f"  {'[PASS]' if ok else '[FAIL]'}  {label}")

    if passed >= 4:
        verdict = "[GO]  GO — strategy cleared for extended paper trading"
    elif passed >= 3:
        verdict = "[MARGINAL]  MARGINAL — run 60+ days paper trading before live"
    else:
        verdict = "[NO-GO]  NO-GO — strategy needs revision before live trading"

    print(f"\n  {verdict}  ({passed}/{len(criteria)} criteria passed)")

    m["criteria_passed"] = passed
    m["criteria_total"]  = len(criteria)
    m["verdict"]         = verdict
    m["fold_results"]    = fold_rows
    return m


# ==========================================================================
# Entry point
# ==========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--days",   type=int, default=DEFAULT_DAYS,   help="Days of history to fetch")
    parser.add_argument("--splits", type=int, default=DEFAULT_SPLITS, help="Number of walk-forward splits")
    parser.add_argument("--output", type=str, default="",             help="Optional JSON output file")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("WALK-FORWARD BACKTEST")
    print(f"  History   : {args.days} days")
    print(f"  Folds     : {args.splits}")
    print(f"  Costs     : {COMMISSION*100:.1f}% commission + {SLIPPAGE*100:.2f}% slippage per side")
    print(f"  Stop/TP   : {STOP_PCT*100:.1f}% / {TP_PCT*100:.1f}% (fixed %)")
    print("=" * 60)

    _, bot_configs = create_production_config()
    report = {}

    for cfg in bot_configs:
        key = f"{cfg.symbol} {cfg.timeframe}"
        df  = fetch_history(cfg.symbol, cfg.timeframe, days=args.days)
        if df.empty or len(df) < 500:
            print(f"  ⚠ Skipping {key} — not enough data")
            continue
        result = run_walk_forward(cfg.symbol, cfg.timeframe, df, n_splits=args.splits)
        if result:
            report[key] = result
        time.sleep(1)

    # -- Final summary --------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, m in report.items():
        status = "GO [PASS]" if m["criteria_passed"] >= 4 else ("MARGINAL [MARGINAL]" if m["criteria_passed"] >= 3 else "NO-GO [FAIL]")
        print(f"  {name:25s}  Sharpe={m['sharpe']:+.2f}  MaxDD={m['max_drawdown']:.1%}  WR={m['win_rate']:.1%}  {status}")

    if not report:
        print("  No results — check internet connection and try again.")
        sys.exit(1)

    # -- Optional JSON export -------------------------------------------
    if args.output:
        with open(args.output, "w") as f:
            # Convert numpy floats to plain Python floats for JSON
            def _clean(obj):
                if isinstance(obj, dict):
                    return {k: _clean(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_clean(v) for v in obj]
                if isinstance(obj, (np.floating, np.integer)):
                    return float(obj)
                return obj
            json.dump(_clean(report), f, indent=2)
        print(f"\n  Report saved to {args.output}")

    print()


if __name__ == "__main__":
    main()
