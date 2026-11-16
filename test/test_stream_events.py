"""Tests for backend.stream (message handling, detectors) and the engine's
event wake-up path. No sockets: handlers are fed recorded-shape messages."""
import asyncio
import csv
import os

import pytest

from backend.agent.engine import AgentConfig, AIAgentEngine
from backend.stream import events as ev
from backend.stream.binance_stream import BinanceStream


def _kline(sym="BTCUSDT", close=100.0, closed=False, t=0):
    return {"stream": f"{sym.lower()}@kline_1m",
            "data": {"e": "kline", "k": {"t": t, "s": sym, "o": close, "h": close, "l": close,
                                         "c": str(close), "v": "1", "x": closed}}}


def _liq(sym="BTCUSDT", side="SELL", price=100.0, qty=10.0):
    return {"e": "forceOrder", "o": {"s": sym, "S": side, "ap": str(price), "z": str(qty)}}


class TestStreamHandlers:
    def test_kline_updates_price_and_closed_bars(self):
        s = BinanceStream(["BTC/USDT"])
        s.handle_spot_message(_kline(close=101.0), now=1000.0)
        assert s.price("BTC/USDT", now=1010.0) == 101.0
        assert s.closes("BTC/USDT") == []
        s.handle_spot_message(_kline(close=102.0, closed=True), now=1020.0)
        assert s.closes("BTC/USDT") == [102.0]

    def test_stale_price_is_none(self):
        s = BinanceStream(["BTC/USDT"])
        s.handle_spot_message(_kline(close=101.0), now=1000.0)
        assert s.price("BTC/USDT", max_age_sec=60, now=1100.0) is None

    def test_unknown_symbol_ignored(self):
        s = BinanceStream(["BTC/USDT"])
        s.handle_spot_message(_kline(sym="DOGEUSDT"), now=1.0)
        assert s.last_price == {}

    def test_liquidations_bucketed_and_written(self, tmp_path):
        s = BinanceStream(["BTC/USDT"], liquidations_dir=str(tmp_path))
        s.handle_liquidation_message(_liq(qty=10.0), now=900.0)       # $1000, older bucket
        s.handle_liquidation_message(_liq(qty=5.0), now=1290.0)       # $500, current bucket
        s.handle_liquidation_message(_liq(sym="XRPUSDT"), now=1290.0)  # other symbol, not counted
        buckets = s.liquidation_buckets("BTC/USDT", bucket_sec=300, n=2, now=1300.0)
        assert buckets == [1000.0, 500.0]
        files = os.listdir(tmp_path)
        assert len(files) == 1
        with open(tmp_path / files[0]) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3 and rows[0]["symbol"] == "BTC/USDT" and rows[0]["side"] == "SELL"

    def test_old_liquidations_pruned(self):
        s = BinanceStream(["BTC/USDT"])
        s.handle_liquidation_message(_liq(), now=0.0)
        s.handle_liquidation_message(_liq(), now=7 * 3600.0)
        assert len(s.liquidations) == 1


class TestDetectors:
    def test_price_move_fires_above_threshold(self):
        closes = [100.0] * 10
        e = ev.price_move("BTC/USDT", closes, 102.0, atr_pct=0.01, window_min=5, atr_mult=1.5)
        assert e and e.kind == "price_move" and "+2.00%" in e.detail

    def test_price_move_quiet_below_threshold(self):
        assert ev.price_move("BTC/USDT", [100.0] * 10, 101.0, 0.01, 5, 1.5) is None

    def test_price_move_needs_history_and_atr(self):
        assert ev.price_move("BTC/USDT", [100.0] * 3, 110.0, 0.01, 5, 1.5) is None
        assert ev.price_move("BTC/USDT", [100.0] * 10, 110.0, None, 5, 1.5) is None

    def test_liquidation_burst(self):
        hist = [100_000.0] * 12
        assert ev.liquidation_burst("BTC/USDT", hist + [400_000.0], 3.0, 250_000.0)
        assert ev.liquidation_burst("BTC/USDT", hist + [200_000.0], 3.0, 250_000.0) is None
        assert ev.liquidation_burst("BTC/USDT", [0.0] * 5 + [1e7], 3.0, 250_000.0) is None  # cold start

    def test_near_stop_long_and_short(self):
        assert ev.near_stop("BTC/USDT", 1, entry=100.0, stop=98.0, price=98.5, fraction_of_r=0.3)
        assert ev.near_stop("BTC/USDT", 1, 100.0, 98.0, 99.5, 0.3) is None
        assert ev.near_stop("BTC/USDT", -1, 100.0, 102.0, 101.5, 0.3)

    def test_fresh_news_only_unseen(self):
        seen = {"u1"}
        out = ev.fresh_news([{"url": "u1", "title": "old"}, {"url": "u2", "title": "new",
                                                             "symbols_mentioned": ["SOL/USDT"]}], seen)
        assert [e.symbol for e in out] == ["SOL/USDT"] and "u2" in seen


class FakeStream:
    def __init__(self):
        self._prices, self._closes, self._liq = {}, {}, {}

    def price(self, symbol, max_age_sec=60.0):
        return self._prices.get(symbol)

    def prices(self, symbols, max_age_sec=60.0):
        return {s: self._prices[s] for s in symbols if s in self._prices}

    def closes(self, symbol):
        return self._closes.get(symbol, [])

    def liquidation_buckets(self, symbol, bucket_sec, n):
        return self._liq.get(symbol, [0.0] * n)


@pytest.fixture
def no_news(monkeypatch):
    monkeypatch.setattr("backend.agent.engine.digest_for_symbols", lambda *a, **k: [])


class TestEngineEvents:
    def _engine(self, mock_llm_client, fake_exchange, stream, **kw):
        return AIAgentEngine(AgentConfig(**kw), exchange=fake_exchange,
                             llm_client=mock_llm_client, stream=stream)

    def test_detects_price_move_from_stream(self, mock_llm_client, fake_exchange, no_news):
        stream = FakeStream()
        stream._closes["BTC/USDT"] = [50000.0] * 10
        stream._prices["BTC/USDT"] = 52000.0          # +4% vs ~1% fake ATR
        engine = self._engine(mock_llm_client, fake_exchange, stream)
        kinds = [(e.kind, e.symbol) for e in engine._detect_events()]
        assert ("price_move", "BTC/USDT") in kinds

    def test_first_news_poll_only_seeds(self, mock_llm_client, fake_exchange, monkeypatch):
        items = [{"url": "a", "title": "A"}]
        monkeypatch.setattr("backend.agent.engine.digest_for_symbols", lambda *a, **k: items)
        engine = self._engine(mock_llm_client, fake_exchange, FakeStream(), news_poll_sec=0)
        assert engine._detect_events() == []
        items.append({"url": "b", "title": "B"})
        assert [e.kind for e in engine._detect_events()] == ["fresh_news"]

    def test_debounce_per_event_key(self, mock_llm_client, fake_exchange):
        engine = self._engine(mock_llm_client, fake_exchange, FakeStream(), event_cooldown_sec=900)
        e = ev.WakeEvent("price_move", "BTC/USDT", "x")
        assert engine._debounced(e, now=1000.0) is False
        assert engine._debounced(e, now=1500.0) is True
        assert engine._debounced(e, now=2000.0) is False

    def test_wait_for_wake_heartbeat_and_event(self, mock_llm_client, fake_exchange):
        engine = self._engine(mock_llm_client, fake_exchange, FakeStream(), min_llm_gap_sec=0)

        async def run():
            engine._wake = asyncio.Queue()
            assert await engine._wait_for_wake(0.01) == []
            a, b = ev.WakeEvent("price_move", "BTC/USDT", "a"), ev.WakeEvent("fresh_news", None, "b")
            engine._wake.put_nowait(a)
            engine._wake.put_nowait(b)
            return await engine._wait_for_wake(1.0)

        assert [e.detail for e in asyncio.run(run())] == ["a", "b"]

    def test_wake_reasons_reach_prompt_and_log(self, mock_llm_client, fake_exchange):
        class Repo:
            calls = []

            def save(self, **kw):
                self.calls.append(kw)

        repo = Repo()
        engine = AIAgentEngine(AgentConfig(), exchange=fake_exchange, llm_client=mock_llm_client,
                               decision_repo=repo, stream=FakeStream())
        asyncio.run(engine._cycle([ev.WakeEvent("liquidation_burst", "SOL/USDT", "SOL liq spike")]))
        assert "SOL liq spike" in mock_llm_client.calls[-1]["messages"][1]["content"]
        assert "SOL liq spike" in repo.calls[-1]["context_snapshot"]

    def test_exit_check_prefers_stream_price(self, mock_llm_client, fake_exchange):
        stream = FakeStream()
        engine = self._engine(mock_llm_client, fake_exchange, stream)
        engine.broker.place_order("BTC/USDT", "buy", 50.0, "t", stop_pct=0.02, target_pct=0.04)
        stream._prices["BTC/USDT"] = 45000.0       # REST (fake_exchange) still says 50000
        asyncio.run(engine._exit_check())
        assert engine.broker.closed_trades[-1].reason == "stop"
