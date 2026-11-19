"""Tournament evidence by event, full-risk cap and judge veto modes."""
import json

import pytest

from backend.agent.llm_client import ChatResult
from backend.allocator.thompson import AllocatorConfig, allocate, cluster_events
from backend.hunters.engine import HunterConfig, HunterEngine
from backend.hunters.judge import run_judge

from test_hunters import FakeBarsExchange, FixedHunter, NOW, _cand, _llm_submitting

MIN = 60_000
DAY = 86_400_000


def _row(t_min, r, direction=1, hunter="a", decision="approved"):
    return {"hunter": hunter, "decision": decision, "r_multiple": r,
            "direction": direction, "created_at": NOW + t_min * MIN}


class TestClusterEvents:
    def test_same_hour_same_direction_is_one_event(self):
        rows = [_row(i * 5, r) for i, r in enumerate([1.0, 2.0, -1.0, 2.0, 1.0])]
        assert cluster_events(rows, 60 * MIN) == [pytest.approx(1.0)]

    def test_opposite_directions_are_separate(self):
        rows = [_row(0, 1.0, 1), _row(5, -1.0, -1)]
        assert sorted(cluster_events(rows, 60 * MIN)) == [-1.0, 1.0]

    def test_window_counts_from_the_first_row_of_the_group(self):
        rows = [_row(0, 1.0), _row(50, 1.0), _row(70, -1.0)]
        assert cluster_events(rows, 60 * MIN) == [1.0, -1.0]

    def test_rows_without_timestamp_stand_alone(self):
        rows = [{"r_multiple": 1.0}, {"r_multiple": -1.0}]
        assert sorted(cluster_events(rows, 60 * MIN)) == [-1.0, 1.0]


class TestFullRiskCap:
    good = [0.8, -0.4] * 20           # 40 events, clearly positive

    def _alloc(self, events, **meta):
        return allocate({"w": events}, {"w": {"live": True, **meta}},
                        AllocatorConfig(capped_scale=0.5))["w"]

    def test_young_record_is_capped(self):
        row = self._alloc(self.good, span_days=3)
        assert row["status"] == "active" and row["risk_scale"] == 0.5 and row["capped"]

    def test_cap_lifts_with_enough_events(self):
        row = self._alloc([0.8, -0.4] * 30, span_days=3)
        assert row["risk_scale"] == 1.0 and not row["capped"]

    def test_cap_lifts_with_enough_days(self):
        row = self._alloc(self.good, span_days=14)
        assert row["risk_scale"] == 1.0 and not row["capped"]

    def test_trades_reported_beside_events(self):
        assert self._alloc(self.good, trades=97)["trades"] == 97


def _engine(**cfg):
    config = HunterConfig(universe=["BTC/USDT"], alloc_explore_scale=1.0, **cfg)
    return HunterEngine(config, FakeBarsExchange(), _llm_submitting(),
                        hunters=[FixedHunter("live_a", ["BTC/USDT"])])


def _veto_all(messages, tools, dispatch, max_tool_turns=6):
    cands = json.loads(messages[1]["content"].split("\n", 1)[1])
    dispatch["submit_verdicts"]({"verdicts": [{"candidate_id": c["candidate_id"], "decision": "veto",
                                               "rationale": "news"} for c in cands]})
    return ChatResult(text="ok")


class TestEngineEvidence:
    def test_cascade_counts_once(self):
        eng = _engine()
        eng._closed_shadow = [_row(i, 1.0, hunter="live_a") for i in range(5)]
        events, info = eng._evidence()
        assert events["live_a"] == [1.0] and info["live_a"]["trades"] == 5
        eng.refresh_allocation(notify=False)
        assert (eng.allocation["live_a"]["n"], eng.allocation["live_a"]["trades"]) == (1, 5)

    def test_span_days(self):
        eng = _engine()
        eng._closed_shadow = [_row(0, 1.0, hunter="live_a"), _row(3 * 24 * 60, 1.0, hunter="live_a")]
        assert eng._evidence()[1]["live_a"]["span_days"] == pytest.approx(3.0)


class TestJudgeVetoModes:
    def test_llm_failure_is_failed_and_not_executed(self):
        eng = _engine(judge_veto="off")
        eng.llm_client = _llm_submitting(error="timeout")
        s = eng.scan_once(now_ms=NOW)
        row = next(iter(eng.shadow.open_rows.values()))
        assert row["decision"] == "failed" and s["executed"] == 0 and eng.broker.open_trades == {}

    def test_ignored_veto_executes_at_half_size(self):
        eng = _engine()                                     # auto, no veto record yet → off
        assert not eng.vetoes_enabled
        eng.llm_client.chat_with_tools = _veto_all
        s = eng.scan_once(now_ms=NOW)
        assert s["executed"] == 1
        row = next(iter(eng.shadow.open_rows.values()))
        assert row["decision"] == "vetoed" and row["live_trade_id"]
        assert row["decision_reason"].startswith("[veto ignored: vetoes disabled]")
        full = min(0.01 * 500 / 0.02, 0.15 * 500)
        assert eng.broker.open_notional()["BTC/USDT"] == pytest.approx(full / 2, rel=0.01)

    def test_obeyed_veto_is_not_executed(self):
        eng = _engine(judge_veto="on")
        eng.llm_client.chat_with_tools = _veto_all
        assert eng.scan_once(now_ms=NOW)["executed"] == 0 and eng.broker.open_trades == {}

    def test_vetoes_count_as_evidence_only_while_ignored(self):
        rows = [_row(i * 120, 1.0, hunter="live_a", decision="vetoed") for i in range(3)]
        off, on = _engine(judge_veto="off"), _engine(judge_veto="on")
        off._closed_shadow = on._closed_shadow = rows
        off.refresh_allocation(notify=False)
        on.refresh_allocation(notify=False)
        assert off.allocation["live_a"]["n"] == 3 and on.allocation["live_a"]["n"] == 0

    def test_auto_reenables_when_vetoes_prove_right_and_emails(self):
        sent = []
        eng = _engine()
        eng.notifier = lambda subject, body: sent.append(body)
        eng._closed_shadow = ([_row(i * 120, -1.0, hunter="live_a", decision="vetoed") for i in range(20)]
                              + [_row(i * 120 + 30, 0.5, hunter="live_a") for i in range(20)])
        eng.refresh_allocation()
        assert eng.vetoes_enabled and any("vetoes ENABLED" in b for b in sent)
        # and they switch off again when the record turns
        eng._closed_shadow = [dict(r, r_multiple=1.0) if r["decision"] == "vetoed" else r
                              for r in eng._closed_shadow]
        eng.refresh_allocation()
        assert not eng.vetoes_enabled and any("vetoes DISABLED" in b for b in sent)

    def test_run_judge_marks_missing_verdict_failed(self):
        a = _cand()
        v, _ = run_judge(_llm_submitting([]), [a], {})
        assert v[a.id] == ("failed", "no verdict returned by judge")
