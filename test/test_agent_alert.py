"""Tests for backend.alerts.agent_alert — pure functions only (no SMTP)."""
import json

from backend.alerts.agent_alert import build_digest, compose_message


def _decision(ts="2026-10-29T23:21:50", cycle_id=1, executed=None, rejected=None,
             halted_reason=None, rationale="held", prompt_tokens=100, completion_tokens=50):
    return {
        "timestamp": ts, "cycle_id": cycle_id,
        "tool_calls_executed": json.dumps(executed or []),
        "guardrail_rejections": json.dumps(rejected or []),
        "halted_reason": halted_reason, "rationale": rationale,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
    }


RUNNING_STATUS = {"running": True, "halted": False, "equity": 500.0, "total_pnl": 0.0,
                  "daily_pnl": 0.0, "positions": []}


class TestBuildDigest:
    def test_empty_day_counts_zero(self):
        digest = build_digest([], RUNNING_STATUS, today="2026-10-29")
        assert digest["cycles_today"] == 0
        assert digest["trades"] == []
        assert digest["kill_switch_hit"] is False

    def test_filters_to_the_given_date(self):
        decisions = [
            _decision(ts="2026-10-29T10:00:00"),
            _decision(ts="2026-10-28T10:00:00"),  # yesterday — excluded
        ]
        digest = build_digest(decisions, RUNNING_STATUS, today="2026-10-29")
        assert digest["cycles_today"] == 1

    def test_hold_counted_as_hold_not_trade(self):
        decisions = [_decision(executed=[{"name": "hold", "output": {"executed": True}}])]
        digest = build_digest(decisions, RUNNING_STATUS, today="2026-10-29")
        assert digest["holds"] == 1
        assert digest["trades"] == []

    def test_executed_order_counted_as_trade(self):
        decisions = [_decision(executed=[{"name": "place_paper_order", "output": {"executed": True}}])]
        digest = build_digest(decisions, RUNNING_STATUS, today="2026-10-29")
        assert len(digest["trades"]) == 1
        assert digest["holds"] == 0

    def test_rejection_counted_separately_from_trade_execution(self):
        decisions = [_decision(
            executed=[{"name": "place_paper_order", "output": {"executed": False}}],
            rejected=[{"name": "place_paper_order", "output": {"rejected_reason": "too big"}}],
        )]
        digest = build_digest(decisions, RUNNING_STATUS, today="2026-10-29")
        assert len(digest["rejections"]) == 1
        assert digest["trades"] == []   # rejected, not executed

    def test_kill_switch_detected(self):
        decisions = [_decision(halted_reason="drawdown_kill_switch")]
        digest = build_digest(decisions, RUNNING_STATUS, today="2026-10-29")
        assert digest["kill_switch_hit"] is True

    def test_non_killswitch_halt_not_flagged_as_kill_switch(self):
        decisions = [_decision(halted_reason="llm_call_budget")]
        digest = build_digest(decisions, RUNNING_STATUS, today="2026-10-29")
        assert digest["kill_switch_hit"] is False
        assert len(digest["halts"]) == 1

    def test_tokens_summed_across_cycles(self):
        decisions = [_decision(prompt_tokens=100, completion_tokens=50),
                    _decision(prompt_tokens=200, completion_tokens=80)]
        digest = build_digest(decisions, RUNNING_STATUS, today="2026-10-29")
        assert digest["tokens"] == 430

    def test_malformed_json_field_does_not_crash(self):
        decisions = [{"timestamp": "2026-10-29T10:00:00", "tool_calls_executed": "{not json",
                     "guardrail_rejections": "{not json either", "halted_reason": None,
                     "rationale": "x", "prompt_tokens": 1, "completion_tokens": 1}]
        digest = build_digest(decisions, RUNNING_STATUS, today="2026-10-29")
        assert digest["cycles_today"] == 1
        assert digest["holds"] == 1   # falls back to "hold" when fields are unparseable


class TestComposeMessage:
    def test_kill_switch_escalates_subject(self):
        digest = build_digest([_decision(halted_reason="drawdown_kill_switch")],
                              {**RUNNING_STATUS, "halted": True}, today="2026-10-29")
        subject, body = compose_message(digest)
        assert "KILL-SWITCH" in subject
        assert "atenção" in body.lower() or "atencao" in body.lower()

    def test_not_running_flagged(self):
        digest = build_digest([], {**RUNNING_STATUS, "running": False}, today="2026-10-29")
        subject, _ = compose_message(digest)
        assert "não está rodando" in subject

    def test_normal_day_subject_mentions_trade_count(self):
        decisions = [_decision(executed=[{"name": "place_paper_order", "output": {"executed": True}}])]
        digest = build_digest(decisions, RUNNING_STATUS, today="2026-10-29")
        subject, body = compose_message(digest)
        assert "1 trade" in subject
        assert "$500" in body or "500.00" in body

    def test_body_lists_rationale_of_trades(self):
        decisions = [_decision(executed=[{"name": "place_paper_order", "output": {"executed": True}}],
                               rationale="bought the dip")]
        digest = build_digest(decisions, RUNNING_STATUS, today="2026-10-29")
        _, body = compose_message(digest)
        assert "bought the dip" in body

    def test_body_lists_rejection_reasons(self):
        decisions = [_decision(
            executed=[{"name": "place_paper_order", "output": {"executed": False}}],
            rejected=[{"name": "place_paper_order", "output": {"rejected_reason": "symbol not allowed"}}],
        )]
        digest = build_digest(decisions, RUNNING_STATUS, today="2026-10-29")
        _, body = compose_message(digest)
        assert "symbol not allowed" in body
