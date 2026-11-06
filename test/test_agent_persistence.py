"""Round-trip tests for the AI agent's dedicated persistence (additive tables,
separate from EquityRepository/DailyStatsRepository — see models.py)."""
from datetime import datetime

from backend.persistence.repository import (
    AgentDailyStatsRepository, AgentDecisionLogRepository, AgentEquityRepository,
)


class TestAgentEquityRepository(object):
    def test_save_and_get_curve(self, fresh_db):
        repo = AgentEquityRepository()
        repo.save(equity=505.0, total_pnl=5.0, daily_pnl=5.0, open_positions=1)
        repo.save(equity=510.0, total_pnl=10.0, daily_pnl=10.0, open_positions=1)
        curve = repo.get_curve(limit=10)
        assert len(curve) == 2
        assert curve[-1]["equity"] == 510.0   # chronological order, newest last

    def test_empty_curve(self, fresh_db):
        assert AgentEquityRepository().get_curve() == []


class TestAgentDailyStatsRepository(object):
    def test_save_creates_row(self, fresh_db):
        repo = AgentDailyStatsRepository()
        repo.save("2026-10-27", pnl=12.5, trades=3, wins=2, losses=1,
                 llm_calls=24, tokens_used=15000)
        history = repo.get_history(limit=10)
        assert len(history) == 1
        assert history[0]["llm_calls"] == 24
        assert history[0]["tokens_used"] == 15000

    def test_save_upserts_same_date(self, fresh_db):
        repo = AgentDailyStatsRepository()
        repo.save("2026-10-27", pnl=5.0, trades=1, wins=1, losses=0)
        repo.save("2026-10-27", pnl=12.5, trades=3, wins=2, losses=1, llm_calls=10)
        history = repo.get_history(limit=10)
        assert len(history) == 1   # upsert, not a second row
        assert history[0]["pnl"] == 12.5
        assert history[0]["llm_calls"] == 10


class TestAgentDecisionLogRepository(object):
    def test_save_and_get_recent(self, fresh_db):
        repo = AgentDecisionLogRepository()
        repo.save(timestamp=datetime.utcnow(), cycle_id=1,
                  symbols_considered='["BTC/USDT"]', raw_llm_output="held",
                  rationale="no clear signal", model_name="gpt-5-mini",
                  prompt_tokens=100, completion_tokens=20, latency_ms=850)
        recent = repo.get_recent(limit=10)
        assert len(recent) == 1
        assert recent[0]["cycle_id"] == 1
        assert recent[0]["rationale"] == "no clear signal"
        assert recent[0]["model_name"] == "gpt-5-mini"

    def test_get_recent_orders_newest_first(self, fresh_db):
        repo = AgentDecisionLogRepository()
        repo.save(timestamp=datetime(2026, 10, 27, 10, 0, 0), cycle_id=1)
        repo.save(timestamp=datetime(2026, 10, 27, 11, 0, 0), cycle_id=2)
        recent = repo.get_recent(limit=10)
        assert recent[0]["cycle_id"] == 2   # newest first

    def test_halted_reason_recorded(self, fresh_db):
        repo = AgentDecisionLogRepository()
        repo.save(timestamp=datetime.utcnow(), cycle_id=1, halted_reason="drawdown_kill_switch")
        recent = repo.get_recent(limit=1)
        assert recent[0]["halted_reason"] == "drawdown_kill_switch"
