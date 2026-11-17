"""
Live Binance market data over WebSocket — the input for event-driven agent
wake-ups and the stop/target monitor.

Two connections:
- spot `<sym>@kline_1m` (combined stream): an update every ~1-2s carries the
  current price, and `x=true` marks a closed 1-minute bar;
- USD-M futures `!forceOrder@arr`: liquidation orders across all symbols.
  Binance sends at most the latest liquidation per symbol per second, so the
  totals undercount real liquidations — fine for spotting relative bursts.
  (Only BTC/ETH/SOL count toward events; every symbol goes to the CSV.)

Liquidations are also appended to a daily CSV (LIQUIDATIONS_DIR), building the
history earlier research never had.

The message handlers are plain methods so they can be unit-tested without a
socket. The connection loops reconnect with exponential backoff and recycle
the socket before Binance's 24h hard cutoff.
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

SPOT_WS = "wss://stream.binance.com:9443/stream?streams="
# USD-M market streams moved to the /market path; the legacy /ws URL stopped
# serving data after 2026-04-23 while still accepting connections.
FUTURES_LIQ_WS = "wss://fstream.binance.com/market/ws/!forceOrder@arr"
_RECYCLE_AFTER_SEC = 23 * 3600
_SPOT_IDLE_SEC = 60           # klines tick every ~1-2s
_LIQ_IDLE_SEC = 1800          # market-wide liquidations can genuinely pause for minutes
_KLINE_HISTORY = 240          # 4h of closed 1m bars per symbol
_LIQ_HISTORY_SEC = 6 * 3600
_LIQ_FIELDS = ["ts", "symbol", "side", "price", "qty", "usd"]


def _ws_symbol(symbol: str) -> str:
    return symbol.replace("/", "").lower()


class BinanceStream:
    def __init__(self, symbols: List[str], liquidations_dir: Optional[str] = None):
        self.symbols = list(symbols)
        self._by_ws = {_ws_symbol(s).upper(): s for s in self.symbols}
        self.liquidations_dir = liquidations_dir
        self.halted = False
        self.last_price: Dict[str, float] = {}
        self.last_price_ts: Dict[str, float] = {}
        self.klines: Dict[str, Deque[dict]] = {s: deque(maxlen=_KLINE_HISTORY) for s in self.symbols}
        self.liquidations: Deque[dict] = deque()
        self.connected = {"spot": False, "liquidations": False}
        self.last_msg_ts: Dict[str, Optional[float]] = {"spot": None, "liquidations": None}
        self.reconnects = {"spot": 0, "liquidations": 0}
        self.liquidations_seen = 0
        self._csv_lock = threading.Lock()

    # ── message handlers (pure-ish, unit-tested) ────────────────────────
    def handle_spot_message(self, msg: dict, now: Optional[float] = None):
        data = msg.get("data", msg)
        k = data.get("k")
        if not k:
            return
        symbol = self._by_ws.get(str(k.get("s", "")).upper())
        if symbol is None:
            return
        now = time.time() if now is None else now
        self.last_price[symbol] = float(k["c"])
        self.last_price_ts[symbol] = now
        if k.get("x"):
            self.klines[symbol].append({"t": int(k["t"]), "o": float(k["o"]), "h": float(k["h"]),
                                        "l": float(k["l"]), "c": float(k["c"]), "v": float(k["v"])})

    def handle_liquidation_message(self, msg: dict, now: Optional[float] = None):
        o = msg.get("o")
        if not o:
            return
        now = time.time() if now is None else now
        raw_sym = str(o.get("s", "")).upper()
        price = float(o.get("ap") or o.get("p") or 0.0)
        qty = float(o.get("z") or o.get("q") or 0.0)
        event = {"ts": now, "symbol": self._by_ws.get(raw_sym, raw_sym),
                 "side": o.get("S"),     # SELL = a long got liquidated, BUY = a short did
                 "price": price, "qty": qty, "usd": price * qty}
        self.liquidations.append(event)
        self.liquidations_seen += 1
        cutoff = now - _LIQ_HISTORY_SEC
        while self.liquidations and self.liquidations[0]["ts"] < cutoff:
            self.liquidations.popleft()
        self._append_csv(event)

    def _append_csv(self, event: dict):
        if not self.liquidations_dir:
            return
        try:
            day = datetime.fromtimestamp(event["ts"], tz=timezone.utc).strftime("%Y-%m-%d")
            path = os.path.join(self.liquidations_dir, f"liquidations_{day}.csv")
            with self._csv_lock:
                new = not os.path.exists(path)
                with open(path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=_LIQ_FIELDS)
                    if new:
                        writer.writeheader()
                    writer.writerow({k: event[k] for k in _LIQ_FIELDS})
        except Exception as exc:
            logger.debug(f"liquidation CSV write failed: {exc}")

    # ── read API ─────────────────────────────────────────────────────────
    def price(self, symbol: str, max_age_sec: float = 60.0, now: Optional[float] = None) -> Optional[float]:
        ts = self.last_price_ts.get(symbol)
        if ts is None or ((time.time() if now is None else now) - ts) > max_age_sec:
            return None
        return self.last_price.get(symbol)

    def prices(self, symbols: List[str], max_age_sec: float = 60.0) -> Dict[str, float]:
        out = {}
        for s in symbols:
            p = self.price(s, max_age_sec)
            if p is not None:
                out[s] = p
        return out

    def closes(self, symbol: str) -> List[float]:
        """Closed 1m bar closes, oldest first."""
        return [b["c"] for b in self.klines.get(symbol, ())]

    def liquidation_buckets(self, symbol: str, bucket_sec: int, n: int,
                            now: Optional[float] = None, side: Optional[str] = None) -> List[float]:
        """USD liquidated on `symbol` per bucket for the last `n` buckets,
        oldest first; the last bucket is the current (partial) one. `side`
        "SELL" counts only liquidated longs, "BUY" only liquidated shorts."""
        now = time.time() if now is None else now
        sums = [0.0] * n
        start = now - bucket_sec * n
        for ev in self.liquidations:
            if ev["symbol"] != symbol or ev["ts"] < start:
                continue
            if side is not None and ev["side"] != side:
                continue
            idx = min(int((ev["ts"] - start) // bucket_sec), n - 1)
            sums[idx] += ev["usd"]
        return sums

    def status(self) -> dict:
        now = time.time()

        def age(ts):
            return round(now - ts, 1) if ts else None

        return {
            "running": not self.halted,
            "connected": dict(self.connected),
            "last_message_age_sec": {k: age(v) for k, v in self.last_msg_ts.items()},
            "reconnects": dict(self.reconnects),
            "price_age_sec": {s: age(self.last_price_ts.get(s)) for s in self.symbols},
            "closed_bars": {s: len(self.klines[s]) for s in self.symbols},
            "liquidations_seen": self.liquidations_seen,
            "liquidations_last_6h": len(self.liquidations),
        }

    # ── connection loops ─────────────────────────────────────────────────
    async def start(self):
        if self.liquidations_dir:
            os.makedirs(self.liquidations_dir, exist_ok=True)
        spot_url = SPOT_WS + "/".join(f"{_ws_symbol(s)}@kline_1m" for s in self.symbols)
        logger.info(f"📡 Binance stream starting ({len(self.symbols)} symbols + liquidations)")
        await asyncio.gather(
            self._run("spot", spot_url, self.handle_spot_message, _SPOT_IDLE_SEC),
            self._run("liquidations", FUTURES_LIQ_WS, self.handle_liquidation_message, _LIQ_IDLE_SEC),
        )

    async def _run(self, name: str, url: str, handler, idle_timeout: float):
        """Reconnect loop. A socket that stays open but silent for
        `idle_timeout` is treated as dead — a deprecated endpoint can accept
        the connection and never send anything."""
        import websockets

        backoff = 1
        while not self.halted:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20,
                                              max_queue=1024) as ws:
                    self.connected[name] = True
                    connected_at = time.time()
                    backoff = 1
                    while True:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=idle_timeout)
                        except asyncio.TimeoutError:
                            raise ConnectionError(f"no message for {idle_timeout:.0f}s")
                        self.last_msg_ts[name] = time.time()
                        try:
                            handler(json.loads(raw))
                        except Exception as exc:
                            logger.debug(f"stream {name}: bad message: {exc}")
                        if self.halted or time.time() - connected_at > _RECYCLE_AFTER_SEC:
                            break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"stream {name} disconnected: {exc}")
            finally:
                self.connected[name] = False
            if self.halted:
                break
            self.reconnects[name] += 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
