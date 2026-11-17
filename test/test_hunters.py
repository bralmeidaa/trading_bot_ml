"""Tests for backend.hunters: causal features, each hunter, the shadow ledger,
the judge (fail-closed) and the engine flow. No network: synthetic bars."""
import json

import numpy as np
import pandas as pd
import pytest

from backend.agent.llm_client import ChatResult
from backend.agent.paper_execution import ROUND_TRIP_COST
from backend.hunters.base import Candidate, Hunter, ScanContext
from backend.hunters.engine import BAR_MS, HunterConfig, HunterEngine
from backend.hunters.features import prepare
from backend.hunters.judge import run_judge
from backend.hunters.shadow import ShadowBook, shadow_stats
from backend.hunters.strategies import (LiquidationReversal, OrbControl, PriorDayFalseBreak,
                                        SessionBreakout, VwapReversion)

DAY0 = int(pd.Timestamp("2026-11-10", tz="UTC").timestamp() * 1000)
CTX = ScanContext(min_stop_pct=4 * ROUND_TRIP_COST, max_stop_pct=0.08)


def bars(closes, start_ms=DAY0, wick=0.002, opens=None, highs=None, lows=None):
    closes = np.asarray(closes, dtype=float)
    o = np.asarray(opens if opens is not None else np.r_[closes[0], closes[:-1]], dtype=float)
    h = np.asarray(highs if highs is not None else np.maximum(o, closes) * (1 + wick))
    lo = np.asarray(lows if lows is not None else np.minimum(o, closes) * (1 - wick))
    ts = start_ms + np.arange(len(closes)) * BAR_MS
    return pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": lo,
                         "close": closes, "volume": 1000.0})


def flat(n, price=100.0, seed=0):
    rng = np.random.default_rng(seed)
    return price * (1 + rng.normal(0, 0.0005, n))


def with_last(df, **cols):
    df = df.copy()
    for k, v in cols.items():
        df.loc[df.index[-1], k] = v
    return df


class TestFeatures:
    def test_features_are_causal(self):
        raw = bars(flat(600) * np.linspace(1, 1.05, 600))
        full = prepare(raw)
        cut = prepare(raw.iloc[:450])
        cols = ["vwap", "vwap_lower", "atr15_pct", "pdh", "pdl", "or_high_ny"]
        pd.testing.assert_frame_equal(full.loc[:449, cols].reset_index(drop=True),
                                      cut[cols].reset_index(drop=True))

    def test_prior_day_levels_and_session_range(self):
        df = prepare(bars(flat(2 * 288 + 200)))
        day2 = df[df["timestamp"] >= DAY0 + 288 * BAR_MS]
        assert day2["pdh"].notna().all()
        ny_rows = df[(df["timestamp"] >= DAY0 + 288 * BAR_MS + 14 * 3_600_000)
                     & (df["timestamp"] < DAY0 + 288 * BAR_MS + 16 * 3_600_000)]
        assert ny_rows["in_window_ny"].all() and ny_rows["or_high_ny"].notna().all()


def _two_days(n_extra=200, price=100.0):
    return flat(2 * 288 + n_extra, price)


class TestHunters:
    def test_vwap_reversion_long_after_flush(self):
        c = list(flat(400))
        c += [99.0, 98.0, 97.0, 96.0, 95.5]                     # flush
        df = bars(c)
        df = with_last(df, open=95.0, low=94.5, high=97.6, close=97.5)   # engulfs the prior bar's high
        cand = VwapReversion().scan("BTC/USDT", prepare(df), CTX)
        assert cand and cand.direction == 1
        assert cand.target_pct >= cand.stop_pct >= CTX.min_stop_pct

    def test_vwap_reversion_quiet_market_no_signal(self):
        assert VwapReversion().scan("BTC/USDT", prepare(bars(flat(400))), CTX) is None

    def test_vwap_reversion_needs_engulfing_turn(self):
        c = list(flat(400)) + [99.0, 98.0, 97.0, 96.0, 95.5]
        weak = with_last(bars(c), open=95.0, low=94.5, high=96.5, close=96.3)   # up bar, but below prior high
        assert VwapReversion().scan("BTC/USDT", prepare(weak), CTX) is None

    def test_session_breakout_only_first_cross(self):
        n = 288 + 14 * 12 + 3
        df = bars(list(flat(n)))
        df.loc[df.index[-3], ["open", "close", "high"]] = [100.0, 102.0, 102.1]   # first break
        df.loc[df.index[-2], ["open", "close", "low"]] = [102.0, 100.0, 99.9]    # back inside
        df.loc[df.index[-1], ["open", "close", "high"]] = [100.0, 102.0, 102.1]   # re-cross
        assert SessionBreakout("ny").scan("BTC/USDT", prepare(df), CTX) is None

    def test_session_breakout_ny_long(self):
        # day 2 up to 14:05 UTC; NY range forms 13:30-14:00 around 100, then a breakout bar
        n = 288 + 14 * 12 + 1
        c = list(flat(n))
        df = bars(c)
        ny_start = DAY0 + 288 * BAR_MS + (13 * 60 + 30) * 60_000
        or_rows = (df["timestamp"] >= ny_start) & (df["timestamp"] < ny_start + 30 * 60_000)
        assert or_rows.sum() == 6
        df.loc[df.index[-1], ["open", "close", "high"]] = [100.0, 102.0, 102.1]
        cand = SessionBreakout("ny").scan("BTC/USDT", prepare(df), CTX)
        assert cand and cand.direction == 1 and cand.target_pct == pytest.approx(2 * cand.stop_pct)

    def test_prior_day_false_break_short(self):
        df = bars(_two_days(100))
        pdh = df.iloc[:288]["high"].max()
        df.loc[df.index[-2], ["close", "high"]] = [pdh * 1.004, pdh * 1.005]
        df.loc[df.index[-1], ["open", "close", "high", "low"]] = [pdh * 1.004, pdh * 0.997, pdh * 1.005, pdh * 0.996]
        cand = PriorDayFalseBreak().scan("BTC/USDT", prepare(df), CTX)
        assert cand and cand.direction == -1

    def test_liquidation_reversal_needs_burst_and_move(self):
        c = list(flat(300)) + [99.0, 98.0, 97.0]
        df = prepare(bars(c))
        burst = ScanContext(CTX.min_stop_pct, CTX.max_stop_pct,
                            liquidations=lambda s, side, b, n: [50_000.0] * 24 + [600_000.0])
        quiet = ScanContext(CTX.min_stop_pct, CTX.max_stop_pct,
                            liquidations=lambda s, side, b, n: [50_000.0] * 25)
        cand = LiquidationReversal().scan("BTC/USDT", df, burst)
        assert cand and cand.direction == 1
        assert LiquidationReversal().scan("BTC/USDT", df, quiet) is None
        assert LiquidationReversal().scan("BTC/USDT", df, CTX) is None   # no stream

    def test_control_is_shadow_only(self):
        assert OrbControl().live is False


def _cand(symbol="BTC/USDT", direction=1, stop=0.02, target=0.04, live=True, hunter="h", ts=1):
    return Candidate(hunter=hunter, version="1", symbol=symbol, direction=direction, bar_ts=ts,
                     ref_price=100.0, stop_pct=stop, target_pct=target, rationale="r", live=live)


class TestShadowBook:
    def test_stop_target_and_time_exits(self):
        book = ShadowBook(max_hold_sec=3600)
        book.open(_cand(ts=1), "approved", "", 100.0, now_ms=0)
        stopped = book.update("BTC/USDT", 1000, price=98.1)              # just past the 98.15 stop
        assert stopped[0]["exit_reason"] == "stop"
        assert -1.2 < stopped[0]["r_multiple"] < -1.0                     # -1R plus exit costs

        book.open(_cand(ts=2), "vetoed", "", 100.0, now_ms=0)
        hit = book.update("BTC/USDT", 1000, price=105.0)
        assert hit[0]["exit_reason"] == "target" and hit[0]["r_multiple"] > 2.0

        book.open(_cand(ts=3), "approved", "", 100.0, now_ms=0)
        assert book.update("BTC/USDT", 1000, price=100.5) == []
        timed = book.update("BTC/USDT", 3_600_000, price=100.5)
        assert timed[0]["exit_reason"] == "time"

    def test_bar_mode_stop_first_and_level_fill(self):
        book = ShadowBook()
        row = book.open(_cand(), "approved", "", 100.0, now_ms=0)
        closed = book.update("BTC/USDT", 1, high=110.0, low=90.0, close=100.0)
        assert closed[0]["exit_reason"] == "stop"
        assert closed[0]["exit_price"] < row["stop_price"]            # exit fill pays costs below the level

    def test_stats_value_add_and_veto_precision(self):
        rows = [
            {"hunter": "a", "decision": "approved", "r_multiple": 2.0},
            {"hunter": "a", "decision": "vetoed", "r_multiple": -1.0},
            {"hunter": "a", "decision": "vetoed", "r_multiple": 1.0},
            {"hunter": "b", "decision": "gated:x", "r_multiple": -1.0},
        ]
        s = shadow_stats(rows)
        assert s["judge"]["taken"]["mean_r"] == 2.0
        assert s["judge"]["judged"]["mean_r"] == pytest.approx(2 / 3, abs=1e-3)
        assert s["judge"]["value_add_r"] == pytest.approx(2.0 - 2 / 3, abs=1e-3)
        assert s["judge"]["veto_precision"] == 0.5
        assert s["hunters"]["b"]["gated"]["n"] == 1


def _llm_submitting(verdicts=None, error=None):
    class _LLM:
        calls = []

        def chat_with_tools(self, messages, tools, dispatch, max_tool_turns=6):
            self.calls.append(messages)
            if error:
                return ChatResult(error=error)
            if verdicts is not None:
                dispatch["submit_verdicts"]({"verdicts": verdicts})
            return ChatResult(text="done", prompt_tokens=100, completion_tokens=20)
    return _LLM()


class TestJudge:
    def test_verdicts_mapped(self):
        a, b = _cand(ts=1), _cand(symbol="ETH/USDT", ts=2)
        llm = _llm_submitting([{"candidate_id": a.id, "decision": "approve", "rationale": "ok"},
                               {"candidate_id": b.id, "decision": "reduce", "rationale": "news risk"}])
        v, _ = run_judge(llm, [a, b], {})
        assert v[a.id] == ("approved", "ok") and v[b.id][0] == "reduced"

    def test_missing_and_unknown_verdicts_fail_closed(self):
        a, b = _cand(ts=1), _cand(symbol="ETH/USDT", ts=2)
        llm = _llm_submitting([{"candidate_id": a.id, "decision": "approve", "rationale": "ok"},
                               {"candidate_id": "made-up", "decision": "approve", "rationale": "x"}])
        v, _ = run_judge(llm, [a, b], {})
        assert v[b.id][0] == "vetoed" and set(v) == {a.id, b.id}

    def test_llm_error_vetoes_everything(self):
        a = _cand()
        v, _ = run_judge(_llm_submitting(error="timeout"), [a], {})
        assert v[a.id][0] == "vetoed" and "timeout" in v[a.id][1]


class FakeBarsExchange:
    def __init__(self, price=100.0):
        self.price = price
        self.n = 0

    def fetch_ohlcv(self, symbol, timeframe="5m", since=None, limit=600):
        self.n += 1
        df = bars(flat(limit, self.price), start_ms=NOW - limit * BAR_MS)
        return df.values.tolist()

    def fetch_ticker(self, symbol):
        return {"last": self.price}

    def fetch_tickers(self, symbols):
        return {s: {"last": self.price} for s in symbols}


NOW = DAY0 + 3 * 288 * BAR_MS


class FixedHunter(Hunter):
    def __init__(self, name, symbols, live=True, direction=1):
        self.name, self.version, self.live = name, "1", live
        self.symbols, self.direction = symbols, direction

    def scan(self, symbol, df, ctx):
        if symbol not in self.symbols:
            return None
        return self._candidate(symbol, df.iloc[-1], self.direction, 0.02, 0.04, f"{self.name} fired")


def _approve_all(messages, tools, dispatch, max_tool_turns=6):
    cands = json.loads(messages[1]["content"].split("\n", 1)[1])
    dispatch["submit_verdicts"]({"verdicts": [{"candidate_id": c["candidate_id"], "decision": "approve",
                                               "rationale": "ok"} for c in cands]})
    return ChatResult(text="ok")


def _engine(hunters, llm, **cfg):
    config = HunterConfig(universe=["BTC/USDT", "ETH/USDT", "SOL/USDT"], **cfg)
    return HunterEngine(config, FakeBarsExchange(), llm, hunters=hunters)


class TestHunterEngine:
    def test_full_flow_shadow_gate_judge_execute(self):
        hunters = [FixedHunter("ctrl", ["BTC/USDT"], live=False),
                   FixedHunter("live_a", ["BTC/USDT", "ETH/USDT"])]
        llm = _llm_submitting()
        eng = _engine(hunters, llm)

        def judge_all(messages, tools, dispatch, max_tool_turns=6):
            cands = json.loads(messages[1]["content"].split("\n", 1)[1])
            dispatch["submit_verdicts"]({"verdicts": [
                {"candidate_id": c["candidate_id"],
                 "decision": "approve" if c["symbol"] == "BTC/USDT" else "veto",
                 "rationale": "test"} for c in cands]})
            return ChatResult(text="ok")
        llm.chat_with_tools = judge_all

        s = eng.scan_once(now_ms=NOW)
        assert s == {"candidates": 3, "shadow_only": 1, "judged": 2, "unfunded": 0, "gated": 0,
                     "executed": 1, "errors": 0}
        assert set(eng.broker.open_trades) == {"BTC/USDT"}
        decisions = sorted(r["decision"] for r in eng.shadow.open_rows.values())
        assert decisions == ["approved", "shadow_only", "vetoed"]
        btc = eng.broker.open_trades["BTC/USDT"]
        assert btc.bot_id == "hunter_judge_v1" and btc.stop_loss > 0
        risk = btc.quantity * abs(btc.entry_price - btc.stop_loss)
        assert risk <= 0.01 * 500 + 1e-6

    def test_open_symbol_is_gated_and_cooldown_suppresses_repeat(self):
        llm = _llm_submitting()
        llm.chat_with_tools = _approve_all
        eng = _engine([FixedHunter("live_a", ["BTC/USDT"])], llm)
        eng.broker.place_order("BTC/USDT", "buy", 20.0, "pre", stop_pct=0.02, target_pct=0.04)
        s = eng.scan_once(now_ms=NOW)
        assert s["judged"] == 1 and s["gated"] == 1 and s["executed"] == 0
        row = next(iter(eng.shadow.open_rows.values()))
        assert row["decision"] == "approved"                     # the verdict is kept as evidence
        assert row["decision_reason"].startswith("[not executed: symbol already has an open position")
        assert eng.scan_once(now_ms=NOW + BAR_MS)["candidates"] == 0       # same hunter+symbol within 1h

    def test_reduce_halves_size_and_budget_blocks_judge(self):
        hunters = [FixedHunter("live_a", ["BTC/USDT"])]
        def reduce_all(messages, tools, dispatch, max_tool_turns=6):
            cid = json.loads(messages[1]["content"].split("\n", 1)[1])[0]["candidate_id"]
            dispatch["submit_verdicts"]({"verdicts": [{"candidate_id": cid, "decision": "reduce", "rationale": "x"}]})
            return ChatResult(text="ok")
        llm = _llm_submitting()
        llm.chat_with_tools = reduce_all
        eng = _engine(hunters, llm, alloc_explore_scale=1.0)
        eng.scan_once(now_ms=NOW)
        notional = eng.broker.open_notional()["BTC/USDT"]
        full = min(0.01 * 500 / 0.02, 0.15 * 500)
        assert notional == pytest.approx(full / 2, rel=0.01)

        eng2 = _engine(hunters, _llm_submitting(), max_llm_calls_per_day=0)
        eng2.scan_once(now_ms=NOW)
        assert [r["decision"] for r in eng2.shadow.open_rows.values()] == ["budget"]
        assert eng2.broker.open_trades == {}

    def test_exit_check_time_stop_and_shadow(self):
        eng = _engine([FixedHunter("live_a", ["BTC/USDT"])], _llm_submitting(), max_hold_sec=60)
        eng.broker.place_order("BTC/USDT", "buy", 20.0, "t", stop_pct=0.02, target_pct=0.04)
        eng.shadow.open(_cand(), "vetoed", "", 100.0, now_ms=0)
        trade = eng.broker.open_trades["BTC/USDT"]
        closed = eng.exit_check_once(now_ms=trade.entry_time + 61_000)
        assert [t.reason for t in closed] == ["time"]
        assert eng.shadow.open_rows == {}                                   # shadow aged out too

    def test_persistence_and_restore(self, fresh_db):
        from backend.persistence.repository import ShadowTradeRepository, TradeRepository
        trade_repo, shadow_repo = TradeRepository(), ShadowTradeRepository()

        def approve_all(messages, tools, dispatch, max_tool_turns=6):
            cands = json.loads(messages[1]["content"].split("\n", 1)[1])
            dispatch["submit_verdicts"]({"verdicts": [{"candidate_id": c["candidate_id"], "decision": "approve",
                                                       "rationale": "ok"} for c in cands]})
            return ChatResult(text="ok")
        llm = _llm_submitting()
        llm.chat_with_tools = approve_all
        cfg = HunterConfig(universe=["BTC/USDT"])
        eng = HunterEngine(cfg, FakeBarsExchange(), llm, trade_repo=trade_repo,
                           shadow_repo=shadow_repo, hunters=[FixedHunter("live_a", ["BTC/USDT"])])
        eng.scan_once(now_ms=NOW)
        assert len(shadow_repo.get_open()) == 1

        again = HunterEngine(cfg, FakeBarsExchange(), llm, trade_repo=trade_repo,
                             shadow_repo=shadow_repo, hunters=[])
        assert again.restore() == 1
        assert set(again.shadow.open_rows) == set(eng.shadow.open_rows)


def test_hunters_endpoint(api_client, fresh_db):
    from backend.persistence.repository import ShadowTradeRepository
    ShadowTradeRepository().save({"id": "x:BTC/USDT:1", "hunter": "x", "version": "1", "symbol": "BTC/USDT",
                                  "direction": 1, "created_at": 1000, "entry_price": 100.0,
                                  "stop_price": 98.0, "target_price": 104.0, "stop_pct": 0.02,
                                  "decision": "approved", "status": "closed", "exit_time": 2000,
                                  "exit_price": 104.0, "exit_reason": "target", "r_multiple": 1.8})
    data = api_client.get("/api/hunters").json()["data"]
    assert data["running"] is False
    assert data["shadow"]["hunters"]["x"]["approved"]["n"] == 1
    rows = api_client.get("/api/hunters/shadow").json()["data"]["rows"]
    assert rows[0]["id"] == "x:BTC/USDT:1"
