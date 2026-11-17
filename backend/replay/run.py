"""
Run the hunters over cached Binance history and report per-strategy results.

    python -m backend.replay.run --days 120
    python -m backend.replay.run --days 60 --symbols BTC/USDT,ETH/USDT --out replay.json

One symbol is loaded, prepared and replayed at a time, so memory stays at a
single symbol's history (the VM has little to spare).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Dict, List, Optional

from backend.hunters.base import Hunter
from backend.replay.report import report
from backend.replay.simulator import ReplayConfig, replay_symbol

logger = logging.getLogger(__name__)

DAY_MS = 86_400_000
WARMUP_DAYS = 3          # feature history before the first signal bar (prior day, VWAP, ATR)


def default_universe() -> List[str]:
    from backend.hunters.engine import HUNTER_UNIVERSE
    return list(HUNTER_UNIVERSE)


def run_replay(hunters: List[Hunter], start_ms: int, end_ms: int,
               symbols: Optional[List[str]] = None, exchange=None,
               cfg: Optional[ReplayConfig] = None, bars_loader=None,
               keep_trades: bool = True) -> dict:
    """Signals from bars with open time in [start_ms, end_ms). `bars_loader`
    (symbol, start_ms, end_ms) -> DataFrame overrides the Binance cache (tests)."""
    symbols = symbols or default_universe()
    if bars_loader is None:
        from backend.data import klines
        if exchange is None:
            import ccxt
            exchange = ccxt.binance({"enableRateLimit": True})

        def bars_loader(sym, a, b):
            return klines.load(exchange, sym, "5m", a, b)

    t0 = time.time()
    trades: List[dict] = []
    coverage: Dict[str, int] = {}
    for sym in symbols:
        try:
            bars = bars_loader(sym, start_ms - WARMUP_DAYS * DAY_MS, end_ms + DAY_MS)
        except Exception as exc:
            logger.warning(f"replay: no data for {sym}: {exc}")
            coverage[sym] = 0
            continue
        coverage[sym] = len(bars)
        # bars after end_ms are only there to let trades opened before it finish
        trades.extend(replay_symbol(sym, bars, hunters, cfg, start_ms, end_ms))
    trades.sort(key=lambda t: t["bar_ts"])
    days = (end_ms - start_ms) / DAY_MS
    out = {
        "start_ms": start_ms, "end_ms": end_ms, "days": round(days, 2),
        "symbols": coverage, "elapsed_sec": round(time.time() - t0, 1),
        "report": report(trades, days, names=[h.name for h in hunters]),
        "notes": ["liquidation_reversal needs the live liquidation feed: never fires in replay",
                  "the LLM judge is not replayed (hindsight); its value is measured live"],
    }
    if keep_trades:
        out["trades"] = trades
    return out


def main():
    from backend.hunters.strategies import default_hunters
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    end = int(time.time() * 1000) // DAY_MS * DAY_MS
    res = run_replay(default_hunters(), end - args.days * DAY_MS, end,
                     symbols=[s for s in args.symbols.split(",") if s] or None)
    for name, rep in res["report"].items():
        print(f"{name:<26} n={rep['n']:<5} mean={rep['mean_r'] if rep['mean_r'] is not None else '-':>8} "
              f"WR={rep['win_rate'] if rep['win_rate'] is not None else '-'} "
              f"folds={rep['folds']['per_fold']} P(>0)={rep['p_positive']} -> {rep['verdict']}")
    print(f"({res['days']}d, {len(res['symbols'])} symbols, {res['elapsed_sec']}s)")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, default=str)


if __name__ == "__main__":
    main()
