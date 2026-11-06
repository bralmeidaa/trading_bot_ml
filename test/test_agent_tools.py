"""Tests for backend.agent.tools.build_tool_dispatch — just the news_tracker
wiring; digest_for_symbols itself is network I/O and not re-tested here."""
from unittest.mock import patch

from backend.agent.paper_execution import PaperBroker
from backend.agent.tools import build_tool_dispatch


class FakeExchange:
    def fetch_ticker(self, symbol):
        return {"last": 100.0, "percentage": 0.0, "high": 100.0, "low": 100.0,
                "quoteVolume": 1_000_000.0}


def _dispatch(news_tracker=None):
    broker = PaperBroker(FakeExchange(), total_capital=500.0)
    return build_tool_dispatch(["BTC/USDT"], FakeExchange(), broker,
                               guardrail_check=lambda payload: (True, None),
                               news_tracker=news_tracker)


class TestNewsTrackerWiring:
    def test_no_tracker_passes_items_through_unmodified(self):
        with patch("backend.agent.tools.digest_for_symbols", return_value=[{"title": "x"}]):
            out = _dispatch()["get_news_digest"]({})
        assert out["news"] == [{"title": "x"}]

    def test_tracker_is_applied_to_items(self):
        def tracker(items):
            return [{**item, "cycles_since_first_seen": 0} for item in items]

        with patch("backend.agent.tools.digest_for_symbols", return_value=[{"title": "x"}]):
            out = _dispatch(news_tracker=tracker)["get_news_digest"]({})
        assert out["news"] == [{"title": "x", "cycles_since_first_seen": 0}]
