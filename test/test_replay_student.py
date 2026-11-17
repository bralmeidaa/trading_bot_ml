"""Phase 4 (replay simulator, klines cache, tournament allocator) and Phase 5
(LLM student: variant specs, validation, lessons). No network."""
import json

import numpy as np
import pandas as pd
import pytest

from backend.agent.llm_client import ChatResult
from backend.agent.paper_execution import COMMISSION, SLIPPAGE
from backend.allocator.thompson import AllocatorConfig, allocate, cash_weight, p_positive, thompson_weights
from backend.data import klines
from backend.hunters.engine import BAR_MS, HunterConfig, HunterEngine
from backend.hunters.judge import JUDGE_PROMPT, run_judge, system_prompt
from backend.hunters.strategies import (BASE_HUNTERS, OrbControl, PriorDayFalseBreak,
                                        SessionBreakout, VwapReversion)
from backend.replay.report import fold_consistency, max_drawdown_r, report, strategy_report
from backend.replay.simulator import ReplayConfig, replay_symbol, simulate_trade
from backend.student.student import Student, StudentConfig, evaluate, verify_lesson
from backend.student.variants import VariantHunter, passes, spec_hash, validate_spec

from test_hunters import DAY0, FakeBarsExchange, FixedHunter, NOW, _approve_all, _llm_submitting, bars

SIDE = COMMISSION + SLIPPAGE


def arr(*xs):
    return [np.asarray(x, dtype=float) for x in xs]


# ── simulator ────────────────────────────────────────────────────────────
class TestSimulateTrade:
    def test_entry_is_next_open_with_costs_and_target(self):
        o, h, l, c = arr([100, 101, 101, 101], [100, 101.5, 106, 101], [100, 100.5, 100.5, 100], [100, 101, 101, 101])
        res = simulate_trade(o, h, l, c, 0, 1, 0.02, 0.04, 10)
        assert res["entry"] == pytest.approx(101 * (1 + SIDE))
        assert res["exit_reason"] == "target" and res["bars_held"] == 2
        target = res["entry"] * 1.04
        assert res["exit"] == pytest.approx(target * (1 - SIDE))
        assert 1.5 < res["r"] < 2.0                         # 2R gross minus costs

    def test_stop_first_when_both_touched(self):
        o, h, l, c = arr([100, 100], [100, 110], [100, 90], [100, 100])
        res = simulate_trade(o, h, l, c, 0, 1, 0.02, 0.04, 10)
        assert res["exit_reason"] == "stop" and res["r"] < -1

    def test_gap_through_stop_fills_at_open(self):
        o, h, l, c = arr([100, 100, 95], [100, 100.5, 96], [100, 99.5, 94], [100, 100, 95])
        res = simulate_trade(o, h, l, c, 0, 1, 0.02, 0.04, 10)
        assert res["exit_reason"] == "stop"
        assert res["exit"] == pytest.approx(95 * (1 - SIDE))  # worse than the stop level
        assert res["r"] < -2

    def test_short_target(self):
        o, h, l, c = arr([100, 100, 99], [100, 100.5, 99.5], [100, 99.5, 95], [100, 100, 96])
        res = simulate_trade(o, h, l, c, 0, -1, 0.02, 0.04, 10)
        assert res["exit_reason"] == "target" and res["r"] > 1.5

    def test_time_exit_and_data_end(self):
        flat = [100.0] * 6
        o, h, l, c = arr(flat, flat, flat, flat)
        res = simulate_trade(o, h, l, c, 0, 1, 0.02, 0.04, 3)
        assert res["exit_reason"] == "time" and res["bars_held"] == 3
        assert res["r"] == pytest.approx(-(2 * SIDE) / 0.02, rel=0.05)   # flat → lose the costs
        assert simulate_trade(o, h, l, c, 0, 1, 0.02, 0.04, 10) is None   # data ends first
        assert simulate_trade(o, h, l, c, 5, 1, 0.02, 0.04, 3) is None


def _volatile(n=4 * 288, seed=3):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.004, n)
    rets[rng.integers(300, n, 25)] += rng.choice([-1, 1], 25) * 0.03        # flushes and spikes
    closes = 100 * np.exp(np.cumsum(rets))
    opens = np.r_[closes[0], closes[:-1]]
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.002, n)))
    return bars(closes, opens=opens, highs=highs, lows=lows)


class _NoShortcut:
    """Wraps a hunter, hiding its prefilter: the replay must give the same trades."""
    def __init__(self, h):
        self.h = h
        self.name, self.version, self.live, self.parent = h.name, h.version, h.live, None

    def prefilter(self, df):
        return None

    def scan_batch(self, *a):
        return None

    def scan(self, *a):
        return self.h.scan(*a)


class TestReplay:
    def test_prefilter_never_changes_results(self):
        raw = _volatile()
        hunters = [VwapReversion().with_params(stretch_atr=2.5), PriorDayFalseBreak(),
                   SessionBreakout("london"), SessionBreakout("ny")]
        fast = replay_symbol("BTC/USDT", raw, hunters)
        slow = replay_symbol("BTC/USDT", raw, [_NoShortcut(h) for h in hunters])
        key = lambda t: (t["hunter"], t["bar_ts"])
        assert len(fast) > 0
        assert sorted(map(key, fast)) == sorted(map(key, slow))
        assert sorted(t["r"] for t in fast) == pytest.approx(sorted(t["r"] for t in slow))

    def test_window_bounds_and_cooldown(self):
        raw = _volatile()
        start = int(raw["timestamp"].iloc[2 * 288])
        end = int(raw["timestamp"].iloc[3 * 288])
        trades = replay_symbol("BTC/USDT", raw, [PriorDayFalseBreak()], start_ms=start, end_ms=end)
        assert all(start <= t["bar_ts"] < end for t in trades)
        per = {}
        for t in sorted(trades, key=lambda t: t["bar_ts"]):
            prev = per.get(t["hunter"])
            assert prev is None or t["bar_ts"] - prev >= ReplayConfig().cooldown_bars * BAR_MS
            per[t["hunter"]] = t["bar_ts"]

    def test_control_uses_batch_path(self):
        raw = _volatile()
        trades = replay_symbol("BTC/USDT", raw, [OrbControl()])
        assert all(t["hunter"] == "orb_control" for t in trades)

    def test_candidates_carry_standard_context(self):
        trades = replay_symbol("BTC/USDT", _volatile(), [PriorDayFalseBreak()])
        assert trades and {"hour", "atr15_pct", "ret_4h"} <= set(trades[0]["context"])


class TestReport:
    def _t(self, rs, hunter="h"):
        return [{"hunter": hunter, "bar_ts": i, "r": r, "direction": 1 if i % 2 else -1,
                 "exit_reason": "target" if r > 0 else "stop"} for i, r in enumerate(rs)]

    def test_verdicts(self):
        assert strategy_report(self._t([1.0, -0.5] * 20), 30)["verdict"] == "GO"
        assert strategy_report(self._t([-0.3] * 40), 30)["verdict"] == "NO-GO"
        assert strategy_report(self._t([1.0] * 10), 30)["verdict"] == "INSUFFICIENT"
        # positive overall but only in one fold → not consistent
        assert strategy_report(self._t([-0.2] * 30 + [3.0] * 10), 30)["verdict"] == "NO-GO"

    def test_folds_drawdown_and_names(self):
        assert fold_consistency([1.0] * 8)["n_folds"] == 0
        assert max_drawdown_r([1, -1, -1, -1, 2]) == -3
        rep = report(self._t([1.0] * 3), 10, names=["h", "silent"])
        assert rep["silent"]["n"] == 0 and rep["h"]["n"] == 3


# ── klines cache ─────────────────────────────────────────────────────────
class CountingExchange:
    def __init__(self):
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        self.calls += 1
        step = klines.TF_MS[timeframe]
        start = since // step * step
        return [[start + k * step, 1, 2, 0.5, 1.5, 10] for k in range(limit)]


class TestKlinesCache:
    def test_months_cached_and_current_month_topped_up(self, tmp_path):
        ex = CountingExchange()
        start = int(pd.Timestamp("2026-09-20", tz="UTC").timestamp() * 1000)
        end = int(pd.Timestamp("2026-10-10", tz="UTC").timestamp() * 1000)
        now = int(pd.Timestamp("2026-10-15", tz="UTC").timestamp() * 1000)
        df = klines.load(ex, "BTC/USDT", "1h", start, end, root=str(tmp_path), now_ms=now)
        assert df["timestamp"].min() >= start and df["timestamp"].max() < end
        assert df["timestamp"].diff().dropna().eq(3_600_000).all()
        assert len(list(tmp_path.iterdir())) == 2
        first = ex.calls
        klines.load(ex, "BTC/USDT", "1h", start, end, root=str(tmp_path), now_ms=now)
        assert ex.calls == first + 1                   # September never again; October topped up
        later = now + 2 * 3_600_000
        df2 = klines.load(ex, "BTC/USDT", "1h", start, later, root=str(tmp_path), now_ms=later)
        assert df2["timestamp"].max() == later - 3_600_000   # only closed bars

    def test_month_iteration(self):
        a = int(pd.Timestamp("2026-11-30", tz="UTC").timestamp() * 1000)
        b = int(pd.Timestamp("2027-01-02", tz="UTC").timestamp() * 1000)
        assert [t for _, _, t in klines.months(a, b)] == ["2026-11", "2026-12", "2027-01"]


# ── allocator ────────────────────────────────────────────────────────────
class TestAllocator:
    def test_posterior_and_thompson(self):
        assert p_positive([]) == pytest.approx(0.5)
        assert p_positive([0.4, -0.2] * 30) > 0.9
        assert p_positive([-0.4, 0.2] * 30) < 0.1
        w = thompson_weights({"good": [0.5, -0.3] * 30, "bad": [-0.5, 0.3] * 30})
        assert w["good"] > 0.8 and w["bad"] < 0.05 and sum(w.values()) == pytest.approx(1.0)

    def test_status_rules(self):
        meta = {"seeded_new": {"live": True}, "variant_new": {"live": True, "variant": True},
                "ctrl": {"live": False}, "loser": {"live": True}, "winner": {"live": True}}
        evid = {"seeded_new": [1.0] * 5, "variant_new": [1.0] * 5, "ctrl": [2.0] * 50,
                "loser": [-0.5, 0.2] * 20, "winner": [0.8, -0.4] * 20}
        a = allocate(evid, meta, AllocatorConfig(explore_scale=0.5))
        assert (a["seeded_new"]["status"], a["seeded_new"]["risk_scale"]) == ("warming", 0.5)
        assert (a["variant_new"]["status"], a["variant_new"]["risk_scale"]) == ("warming", 0.0)
        assert (a["ctrl"]["status"], a["ctrl"]["risk_scale"]) == ("control", 0.0)
        assert (a["loser"]["status"], a["loser"]["risk_scale"]) == ("demoted", 0.0)
        assert a["winner"]["status"] == "active" and a["winner"]["risk_scale"] == 1.0

    def test_weak_edge_leaves_most_in_cash(self):
        a = allocate({"meh": [0.6, -0.55] * 15}, {"meh": {"live": True}})
        assert a["meh"]["status"] == "active"
        assert a["meh"]["risk_scale"] < 1.0 and cash_weight(a) > 0.3


class TestEngineTournament:
    def _engine(self, hunters, **cfg):
        config = HunterConfig(universe=["BTC/USDT"], **cfg)
        return HunterEngine(config, FakeBarsExchange(), _llm_submitting(), hunters=hunters)

    def test_unfunded_is_judged_but_not_executed(self):
        h = FixedHunter("v", ["BTC/USDT"])
        h.parent = "vwap_reversion"                         # a student variant, still warming
        eng = self._engine([h])
        eng.llm_client.chat_with_tools = _approve_all
        eng.refresh_allocation(notify=False)
        s = eng.scan_once(now_ms=NOW)
        assert (s["judged"], s["unfunded"], s["executed"]) == (1, 1, 0)
        row = next(iter(eng.shadow.open_rows.values()))
        assert row["decision"] == "approved" and "no tournament capital (warming)" in row["decision_reason"]
        assert eng.broker.open_trades == {}

    def test_replay_no_go_removes_exploration_capital(self):
        eng = self._engine([FixedHunter("a", ["BTC/USDT"]), FixedHunter("b", ["BTC/USDT"])])
        eng.set_replay_verdicts({"report": {"a": {"verdict": "NO-GO"}, "b": {"verdict": "GO"}}})
        assert eng.allocation["a"]["risk_scale"] == 0.0 and eng.allocation["a"]["replay"] == "NO-GO"
        assert eng.allocation["b"]["risk_scale"] == eng.config.alloc_explore_scale
        # forward evidence of the judge's picks overrides the replay verdict
        eng._closed_shadow = [{"hunter": "a", "decision": "approved", "r_multiple": r}
                              for r in [0.8, -0.4] * 20] + \
                             [{"hunter": "a", "decision": "vetoed", "r_multiple": -5.0}] * 40
        eng.refresh_allocation(notify=False)
        assert eng.allocation["a"]["status"] == "active" and eng.allocation["a"]["n"] == 40

    def test_explore_scale_sizes_seeded_hunters(self):
        def approve(messages, tools, dispatch, max_tool_turns=6):
            cid = json.loads(messages[1]["content"].split("\n", 1)[1])[0]["candidate_id"]
            dispatch["submit_verdicts"]({"verdicts": [{"candidate_id": cid, "decision": "approve",
                                                       "rationale": "ok"}]})
            return ChatResult(text="ok")
        eng = self._engine([FixedHunter("a", ["BTC/USDT"])], alloc_explore_scale=0.5)
        eng.llm_client.chat_with_tools = approve
        eng.scan_once(now_ms=NOW)
        full = min(0.01 * 500 / 0.02, 0.15 * 500)
        assert eng.broker.open_notional()["BTC/USDT"] == pytest.approx(full * 0.5, rel=0.01)

    def test_status_change_is_notified(self):
        sent = []
        eng = self._engine([FixedHunter("a", ["BTC/USDT"])])
        eng.notifier = lambda subj, body: sent.append((subj, body))
        eng.refresh_allocation()
        assert eng.allocation["a"]["status"] == "warming" and not sent
        eng._closed_shadow = [{"hunter": "a", "decision": "reduced", "r_multiple": r} for r in [0.8, -0.4] * 20]
        eng.refresh_allocation()
        assert eng.allocation["a"]["status"] == "active"
        assert sent and "warming -> active" in sent[0][1]

    def test_set_variants_keeps_seeded(self):
        eng = self._engine([FixedHunter("a", ["BTC/USDT"])])
        v = VariantHunter("vwap_reversion~v1", {"parent": "vwap_reversion", "params": {"stretch_atr": 3.0},
                                                "filters": []})
        eng.set_variants([v])
        assert [h.name for h in eng.hunters] == ["a", "vwap_reversion~v1"]
        eng.set_variants([])
        assert [h.name for h in eng.hunters] == ["a"]


# ── student: specs ───────────────────────────────────────────────────────
class TestVariantSpecs:
    def test_valid_spec_normalized(self):
        spec, err = validate_spec({"parent": "vwap_reversion", "params": {"stretch_atr": 3.0, "lookback_bars": 24},
                                   "filters": [{"feature": "hour", "op": "between", "value": [13, 20]}],
                                   "rationale": "ny only"})
        assert err is None and spec["params"] == {"stretch_atr": 3.0}     # unchanged param dropped
        assert spec_hash(spec) == spec_hash(dict(spec, rationale="other words"))

    @pytest.mark.parametrize("raw, msg", [
        ({"parent": "nope"}, "unknown parent"),
        ({"parent": "liquidation_reversal", "params": {"burst_mult": 4.0}}, "cannot be validated"),
        ({"parent": "vwap_reversion", "params": {"stretch_atr": 9.0}}, "outside"),
        ({"parent": "vwap_reversion", "params": {"exec": 1}}, "no tunable parameter"),
        ({"parent": "vwap_reversion", "filters": [{"feature": "btc_secret", "op": "<", "value": 1}]}, "not in the allowed"),
        ({"parent": "vwap_reversion", "filters": [{"feature": "hour", "op": "between", "value": [20, 13]}]}, "between"),
        ({"parent": "vwap_reversion", "filters": [{"feature": "hour", "op": "<", "value": 99}]}, "outside its range"),
        ({"parent": "vwap_reversion", "params": {}, "filters": []}, "no change"),
    ])
    def test_invalid_specs(self, raw, msg):
        spec, err = validate_spec(raw)
        assert spec is None and msg in err

    def test_passes_fails_closed(self):
        f = [{"feature": "hour", "op": "between", "value": [13, 20]}, {"feature": "direction", "op": "in", "value": [-1]}]
        assert passes(f, {"hour": 14}, -1)
        assert not passes(f, {"hour": 14}, 1)
        assert not passes(f, {}, -1)                          # missing feature → fail

    def test_variant_hunter_filters_and_relabels(self):
        raw = _volatile()
        from backend.hunters.features import prepare
        df = prepare(raw)
        parent = PriorDayFalseBreak()
        open_all = VariantHunter("prior_day_false_break~v1", {"parent": "prior_day_false_break",
                                                               "params": {"target_r": 3.0}, "filters": []})
        trades_p = replay_symbol("BTC/USDT", raw, [parent])
        trades_v = replay_symbol("BTC/USDT", raw, [open_all])
        assert len(trades_p) == len(trades_v) > 0
        assert all(t["hunter"] == "prior_day_false_break~v1" for t in trades_v)
        assert all(t["target_pct"] == pytest.approx(3 * t["stop_pct"]) for t in trades_v)
        longs = VariantHunter("prior_day_false_break~v2", {"parent": "prior_day_false_break", "params": {},
                                                            "filters": [{"feature": "direction", "op": "in", "value": [1]}]})
        assert all(t["direction"] == 1 for t in replay_symbol("BTC/USDT", raw, [longs]))

    def test_with_params_enforces_bounds(self):
        with pytest.raises(ValueError):
            VwapReversion().with_params(stretch_atr=100)
        assert VwapReversion().with_params(lookback_bars=30.0).lookback_bars == 30


# ── student: the loop ────────────────────────────────────────────────────
def _trades(hunter, rs, ctx=None):
    return [{"hunter": hunter, "bar_ts": i, "r": r, "direction": 1, "exit_reason": "x",
             "context": ctx or {"hour": 14}} for i, r in enumerate(rs)]


GOOD = [1.0, -0.5] * 20            # +0.25R, consistent
BAD = [-0.6, 0.2] * 20             # -0.2R


def fake_replay(calls):
    table = {"vwap_reversion": BAD, "vwap_reversion~v1": GOOD, "vwap_reversion~v2": BAD,
             "prior_day_false_break": BAD}

    def _run(hunters, start, end):
        calls.append(([h.name for h in hunters], start, end))
        out = []
        for h in hunters:
            out += _trades(h.name, table.get(h.name, [0.1] * 5))
        return out
    return _run


def student_llm(hypotheses, lessons=()):
    class _LLM:
        def __init__(self):
            self.messages = None

        def chat_with_tools(self, messages, tools, dispatch, max_tool_turns=3):
            self.messages = messages
            dispatch["submit_proposals"]({"hypotheses": list(hypotheses), "lessons": list(lessons)})
            return ChatResult(text="ok", prompt_tokens=100, completion_tokens=50)
    return _LLM()


def _shadow_row(i, hunter, r, ctx, decision="approved"):
    return {"id": f"{hunter}:BTC/USDT:{i}", "hunter": hunter, "version": "1", "symbol": "BTC/USDT",
            "direction": -1, "created_at": 1000 + i, "entry_price": 100.0, "stop_price": 102.0,
            "target_price": 96.0, "stop_pct": 0.02, "decision": decision, "context": json.dumps(ctx),
            "status": "closed", "exit_time": 2000 + i, "exit_price": 100.0, "exit_reason": "stop",
            "r_multiple": r}


@pytest.fixture
def repos(fresh_db):
    from backend.persistence.repository import (LessonRepository, ShadowTradeRepository,
                                                StrategyVariantRepository)
    return ShadowTradeRepository(), StrategyVariantRepository(), LessonRepository()


NOW_S = DAY0 / 1000 + 200 * 86400


class TestStudent:
    def test_evaluate_rules(self):
        cfg = StudentConfig()
        ok, why, _ = evaluate(_trades("v", GOOD), _trades("p", BAD), 40, cfg)
        assert ok and "out of sample" in why
        assert not evaluate(_trades("v", BAD), _trades("p", BAD), 40, cfg)[0]
        assert "only 10 trades" in evaluate(_trades("v", [1.0] * 10), [], 40, cfg)[1]
        # positive but not better than an even better parent
        assert "does not beat" in evaluate(_trades("v", GOOD), _trades("p", [1.2, -0.5] * 20), 40, cfg)[1]

    def test_weekly_good_promoted_bad_rejected(self, repos):
        shadow_repo, var_repo, les_repo = repos
        sent, calls = [], []
        hyps = [{"parent": "vwap_reversion", "params": {"stretch_atr": 3.0}, "rationale": "looser stretch"},
                {"parent": "vwap_reversion", "filters": [{"feature": "hour", "op": ">", "value": 12}],
                 "rationale": "afternoon only"},
                {"parent": "vwap_reversion", "params": {"stretch_atr": 3.0}, "rationale": "same again"},
                {"parent": "vwap_reversion", "params": {"stretch_atr": 99}, "rationale": "too far"}]
        llm = student_llm(hyps)
        eng = HunterEngine(HunterConfig(universe=["BTC/USDT"]), FakeBarsExchange(), _llm_submitting(),
                           hunters=[VwapReversion()])
        st = Student(llm, shadow_repo, var_repo, les_repo, engine=eng,
                     notifier=lambda s, b: sent.append((s, b)), replay_fn=fake_replay(calls),
                     now_fn=lambda: NOW_S)
        out = st.weekly()
        statuses = {h.get("id", h["spec"].get("rationale")): h["status"] for h in out["hypotheses"]}
        assert statuses["vwap_reversion~v1"] == "shadow"
        assert statuses["vwap_reversion~v2"] == "rejected"
        assert sorted(statuses.values()).count("duplicate") == 1
        assert sorted(statuses.values()).count("invalid") == 1

        # the dossier came from the TRAIN segment; validation ran on the later TEST segment
        (train_names, a, b), (test_names, c, d) = calls
        assert b == c and a < b < d and "vwap_reversion~v1" in test_names
        dossier = json.loads(llm.messages[1]["content"])
        assert "vwap_reversion" in dossier["replay_training_segment"]["per_hunter"]
        assert "vwap_reversion~v1" not in json.dumps(dossier["replay_training_segment"])

        assert any("new variant" in s for s, _ in sent)
        assert [v["status"] for v in var_repo.list()] == ["shadow", "rejected"]
        assert "vwap_reversion~v1" in [h.name for h in eng.hunters]      # applied, with zero capital
        assert eng.allocation["vwap_reversion~v1"]["risk_scale"] == 0.0

    def test_lessons_need_shadow_evidence(self, repos):
        shadow_repo, var_repo, les_repo = repos
        for i in range(34):
            shadow_repo.save(_shadow_row(i, "vwap_reversion", -0.6 if i % 4 else 0.3, {"hour": 2}))
        for i in range(34, 44):
            shadow_repo.save(_shadow_row(i, "vwap_reversion", 0.5, {"hour": 15}))
        lessons = [{"text": "VWAP reversion in Asia hours loses", "hunter": "vwap_reversion",
                    "filters": [{"feature": "hour", "op": "<", "value": 7}], "claim": "loses"},
                   {"text": "VWAP reversion in the afternoon wins", "hunter": "vwap_reversion",
                    "filters": [{"feature": "hour", "op": ">", "value": 12}], "claim": "wins"}]
        eng = HunterEngine(HunterConfig(universe=["BTC/USDT"]), FakeBarsExchange(), _llm_submitting(),
                           hunters=[VwapReversion()])
        st = Student(student_llm([], lessons), shadow_repo, var_repo, les_repo, engine=eng,
                     replay_fn=fake_replay([]), now_fn=lambda: NOW_S)
        out = st.weekly()
        assert [l["status"] for l in out["lessons"]] == ["active", "unsupported"]   # 10 cases < 30
        active = les_repo.list(status="active")
        assert len(active) == 1 and "[n=34" in active[0]["text"]
        assert eng.lessons == [active[0]["text"]]
        assert active[0]["text"] in system_prompt(eng.lessons) and system_prompt([]) == JUDGE_PROMPT

    def test_daily_retires_variant_losing_forward(self, repos):
        shadow_repo, var_repo, les_repo = repos
        var_repo.save({"id": "vwap_reversion~v1", "parent": "vwap_reversion", "spec_hash": "x",
                       "params": {"stretch_atr": 3.0}, "filters": [], "status": "shadow"})
        for i in range(40):
            shadow_repo.save(_shadow_row(i, "vwap_reversion~v1", -0.7 if i % 3 else 0.4, {}, "approved"))
        sent = []
        st = Student(None, shadow_repo, var_repo, les_repo, notifier=lambda s, b: sent.append(s),
                     replay_fn=fake_replay([]))
        out = st.daily()
        assert out["retired_variants"] and var_repo.list()[0]["status"] == "retired"
        assert sent and "retired 1 variant" in sent[0]

    def test_verify_lesson_counts_only_matching(self):
        trades = [{"hunter": "a", "direction": 1, "r": -0.5, "context": {"hour": 2}}] * 31 + \
                 [{"hunter": "b", "direction": 1, "r": -0.5, "context": {"hour": 2}}] * 5
        chk = verify_lesson({"hunter": "a", "filters": [{"feature": "hour", "op": "<", "value": 7}],
                             "claim": "loses"}, trades)
        assert chk["n"] == 31 and chk["confidence"] > 0.99


def test_judge_receives_lessons():
    class _LLM:
        def chat_with_tools(self, messages, tools, dispatch, max_tool_turns=6):
            self.system = messages[0]["content"]
            return ChatResult(text="")
    llm = _LLM()
    from backend.hunters.base import Candidate
    c = Candidate("h", "1", "BTC/USDT", 1, 0, 100.0, 0.02, 0.04, "x")
    run_judge(llm, [c], {}, lessons=["Asia loses [n=31, -0.40R/trade]"])
    assert "Asia loses" in llm.system


def test_new_endpoints(api_client, fresh_db):
    t = api_client.get("/api/tournament").json()["data"]
    assert "vwap_reversion" in t["strategies"] and t["strategies"]["orb_control"]["status"] == "control"
    s = api_client.get("/api/student").json()["data"]
    assert s["variants"] == [] and s["lessons"] == []
    r = api_client.get("/api/replay").json()["data"]
    assert r["running"] is False
    assert api_client.post("/api/student/run?mode=bogus").status_code == 400
    assert api_client.post("/api/replay/run?days=2").status_code == 400
