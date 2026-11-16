"""API tests for the agent's trade ledger endpoint (no agent running)."""
from backend.agent.paper_execution import AgentTrade


def test_agent_trades_endpoint_reports_stats(api_client, fresh_db):
    from backend.persistence.repository import TradeRepository
    repo = TradeRepository()
    repo.save(AgentTrade(id="t1", symbol="BTC/USDT", direction=1, entry_time=1_000, entry_price=100.0,
                         quantity=1.0, stop_loss=98.0, take_profit=104.0, exit_time=2_000,
                         exit_price=104.0, pnl=4.0, pnl_pct=0.04, status="closed",
                         reason="target", bot_id="ai_agent_v2"))
    repo.save(AgentTrade(id="t2", symbol="ETH/USDT", direction=-1, entry_time=3_000, entry_price=50.0,
                         quantity=1.0, stop_loss=51.0, take_profit=48.0, bot_id="ai_agent_v2"))
    repo.save(AgentTrade(id="old", symbol="BTC/USDT", direction=1, entry_time=1, entry_price=1.0,
                         quantity=1.0, stop_loss=0.0, take_profit=0.0, exit_time=2, exit_price=2.0,
                         pnl=1.0, pnl_pct=1.0, status="closed", bot_id="ai_agent"))

    data = api_client.get("/api/agent/trades").json()["data"]
    assert data["bot_id"] == "ai_agent_v2"
    assert data["stats"]["n"] == 1 and data["stats"]["net_pnl"] == 4.0
    assert data["stats"]["expectancy_r"] == 2.0
    assert [t["id"] for t in data["open"]] == ["t2"]
