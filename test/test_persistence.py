"""
Database persistence tests — verify that trades, equity snapshots,
and daily stats are correctly saved and retrieved.

Uses an in-memory SQLite DB via the DATABASE_URL env var set in conftest.py.
"""
import pytest
from datetime import datetime


def _make_mock_trade(trade_id="t1", pnl=50.0, pnl_pct=0.03, bot_id="LINK/USDT_5m"):
    """Return a simple namespace mimicking a Trade dataclass."""
    from types import SimpleNamespace
    return SimpleNamespace(
        id=trade_id,
        bot_id=bot_id,
        symbol="LINK/USDT",
        direction=1,
        entry_time=int(datetime.now().timestamp() * 1000),
        exit_time=int(datetime.now().timestamp() * 1000) + 300_000,
        entry_price=15.0,
        exit_price=15.45,
        quantity=10.0,
        stop_loss=14.5,
        take_profit=16.0,
        pnl=pnl,
        pnl_pct=pnl_pct,
        status="closed",
        reason="take_profit",
    )


class TestTradeRepository:
    def test_save_and_retrieve(self, fresh_db):
        from backend.persistence.repository import TradeRepository
        repo = TradeRepository()
        trade = _make_mock_trade("tr_1", pnl=50.0, pnl_pct=0.03)

        repo.save(trade)
        results = repo.get_recent_trades(limit=10)

        assert len(results) == 1
        assert results[0]["id"] == "tr_1"
        assert results[0]["pnl"] == 50.0

    def test_upsert_updates_existing(self, fresh_db):
        from backend.persistence.repository import TradeRepository
        repo = TradeRepository()
        trade = _make_mock_trade("tr_dup", pnl=10.0, pnl_pct=0.01)
        repo.save(trade)

        trade.pnl = 99.0  # update
        repo.save(trade)

        results = repo.get_recent_trades()
        assert len(results) == 1
        assert results[0]["pnl"] == 99.0

    def test_get_recent_pnl_pcts(self, fresh_db):
        from backend.persistence.repository import TradeRepository
        repo = TradeRepository()
        for i in range(5):
            repo.save(_make_mock_trade(f"tr_{i}", pnl_pct=i * 0.01))

        pnl_pcts = repo.get_recent_pnl_pcts(limit=10)
        assert len(pnl_pcts) == 5
        assert all(isinstance(v, float) for v in pnl_pcts)

    def test_count(self, fresh_db):
        from backend.persistence.repository import TradeRepository
        repo = TradeRepository()
        assert repo.count() == 0
        repo.save(_make_mock_trade("tr_c1"))
        assert repo.count() == 1

    def test_oldest_first_ordering(self, fresh_db):
        """get_recent_pnl_pcts should return in chronological order."""
        import time
        from backend.persistence.repository import TradeRepository
        repo = TradeRepository()
        repo.save(_make_mock_trade("early", pnl_pct=0.01))
        time.sleep(0.01)
        repo.save(_make_mock_trade("late", pnl_pct=0.05))

        pcts = repo.get_recent_pnl_pcts(limit=10)
        assert pcts[0] == pytest.approx(0.01)   # oldest first
        assert pcts[-1] == pytest.approx(0.05)  # newest last


class TestEquityRepository:
    def test_save_and_retrieve(self, fresh_db):
        from backend.persistence.repository import EquityRepository
        repo = EquityRepository()
        repo.save(equity=10100.0, total_pnl=100.0, daily_pnl=50.0, active_trades=2)

        curve = repo.get_curve(limit=10)
        assert len(curve) == 1
        assert curve[0]["equity"] == 10100.0
        assert curve[0]["total_pnl"] == 100.0
        assert curve[0]["active_trades"] == 2

    def test_multiple_snapshots_in_order(self, fresh_db):
        from backend.persistence.repository import EquityRepository
        import time
        repo = EquityRepository()
        for equity in [10000.0, 10050.0, 10100.0]:
            repo.save(equity=equity, total_pnl=equity - 10000, daily_pnl=0)
            time.sleep(0.01)

        curve = repo.get_curve(limit=10)
        assert len(curve) == 3
        assert curve[0]["equity"] == 10000.0  # oldest first after reverse


class TestDailyStatsRepository:
    def test_save_and_retrieve(self, fresh_db):
        from backend.persistence.repository import DailyStatsRepository
        repo = DailyStatsRepository()
        repo.save("2025-06-01", pnl=120.0, trades=5, wins=3, losses=2)

        history = repo.get_history(limit=30)
        assert len(history) == 1
        assert history[0]["date"] == "2025-06-01"
        assert history[0]["pnl"] == 120.0
        assert history[0]["wins"] == 3

    def test_upsert_by_date(self, fresh_db):
        from backend.persistence.repository import DailyStatsRepository
        repo = DailyStatsRepository()
        repo.save("2025-06-01", pnl=100.0, trades=3, wins=2, losses=1)
        repo.save("2025-06-01", pnl=200.0, trades=6, wins=4, losses=2)  # same date

        history = repo.get_history()
        assert len(history) == 1
        assert history[0]["pnl"] == 200.0  # updated

    def test_multiple_days(self, fresh_db):
        from backend.persistence.repository import DailyStatsRepository
        repo = DailyStatsRepository()
        for day in ["2025-06-01", "2025-06-02", "2025-06-03"]:
            repo.save(day, pnl=10.0, trades=1, wins=1, losses=0)

        history = repo.get_history(limit=5)
        assert len(history) == 3
