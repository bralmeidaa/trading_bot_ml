"""
Live order book collector — the parallel data-accumulation track.

Periodically snapshots L2 order book per symbol and stores COMPACT microstructure
features (spread, imbalance, depth, microprice) to daily CSV files. This builds
proprietary history of the one input the model never had (order flow), so we can
test microstructure-based intraday signals in the future (no historical L2 exists
to backtest, so we must collect it going forward).

Runs alongside the portfolio engine. Pure feature computation (compute_features)
is unit-tested without network.
"""
from __future__ import annotations

import asyncio
import csv
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_FIELDS = ["timestamp", "symbol", "mid", "spread_bps", "imbalance_top5",
           "imbalance_top20", "bid_vol_top20", "ask_vol_top20", "microprice"]


def compute_features(ob: dict, symbol: str, ts: int) -> Optional[dict]:
    """Compact microstructure features from a CCXT order book dict.

    ob = {"bids": [[price, amount], ...], "asks": [[price, amount], ...]}
    Returns None if the book is empty/degenerate.
    """
    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    if not bids or not asks:
        return None
    best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
    if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
        return None
    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 1e4

    def _imb(n):
        b = sum(float(a) for _, a in bids[:n])
        a = sum(float(x) for _, x in asks[:n])
        tot = b + a
        return (b - a) / tot if tot > 0 else 0.0, b, a

    imb5, _, _ = _imb(5)
    imb20, bid_vol20, ask_vol20 = _imb(20)

    # microprice: weighted toward the side with the larger OPPOSITE top queue
    q_bid, q_ask = float(bids[0][1]), float(asks[0][1])
    micro = ((best_bid * q_ask + best_ask * q_bid) / (q_bid + q_ask)
             if (q_bid + q_ask) > 0 else mid)

    return {
        "timestamp": ts, "symbol": symbol,
        "mid": round(mid, 8), "spread_bps": round(spread_bps, 3),
        "imbalance_top5": round(imb5, 4), "imbalance_top20": round(imb20, 4),
        "bid_vol_top20": round(bid_vol20, 4), "ask_vol_top20": round(ask_vol20, 4),
        "microprice": round(micro, 8),
    }


class OrderBookCollector:
    def __init__(self, symbols: List[str], exchange=None, interval_sec: int = 60,
                 depth: int = 20, storage_dir: str = "orderbook_data"):
        self.symbols = symbols
        self.exchange = exchange
        self.interval_sec = interval_sec
        self.depth = depth
        self.storage_dir = storage_dir
        self.halted = False
        self.snapshots = 0
        self.last_ts = 0
        self._lock = threading.Lock()
        os.makedirs(storage_dir, exist_ok=True)

    # ── storage ─────────────────────────────────────────────────────────
    def _path_for_today(self) -> str:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self.storage_dir, f"orderbook_{day}.csv")

    def append(self, feat: dict):
        path = self._path_for_today()
        with self._lock:
            new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(feat)

    # ── live loop ───────────────────────────────────────────────────────
    async def start(self):
        logger.info(f"📡 Order book collector starting — {len(self.symbols)} symbols "
                    f"every {self.interval_sec}s → {self.storage_dir}/")
        try:
            while not self.halted:
                await self._cycle()
                await asyncio.sleep(self.interval_sec)
        except asyncio.CancelledError:
            logger.info("Order book collector cancelled")
        except Exception as exc:
            logger.error(f"Order book collector error: {exc}")

    async def _cycle(self):
        if self.exchange is None:
            return
        ts = int(time.time() * 1000)
        for sym in self.symbols:
            try:
                ob = await asyncio.to_thread(self.exchange.fetch_order_book, sym, self.depth)
                feat = compute_features(ob, sym, ts)
                if feat:
                    self.append(feat)
                    self.snapshots += 1
                    self.last_ts = ts
            except Exception as exc:
                logger.debug(f"collector {sym}: {exc}")
            await asyncio.sleep(0.2)   # gentle rate limiting

    def status(self) -> dict:
        return {
            "running": not self.halted,
            "symbols": len(self.symbols),
            "interval_sec": self.interval_sec,
            "snapshots_written": self.snapshots,
            "last_snapshot": self.last_ts,
            "storage_dir": self.storage_dir,
            "today_file": os.path.basename(self._path_for_today()),
        }
