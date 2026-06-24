"""
API endpoint tests — verify every route works and follows the {success, data} contract.

All successful responses are {"success": true, "data": {...}}.
The `unwrap()` helper asserts that envelope and returns the inner data.
Errors (HTTPException) keep FastAPI's {"detail": ...} format and proper status codes.

Tests run against a TestClient with no live trading system (demo data mode),
except TestBotManagementWithSystem which injects a fake system.
"""
import pytest


def unwrap(response, expected_status=200):
    """Assert the {success, data} envelope and return the inner data payload."""
    assert response.status_code == expected_status, response.text
    body = response.json()
    assert body.get("success") is True, f"Expected success=true, got: {body}"
    assert "data" in body, f"Missing 'data' key in: {body}"
    return body["data"]


class TestCoreEndpoints:
    def test_health_check(self, api_client):
        data = unwrap(api_client.get("/api/health"))
        assert data["status"] == "healthy"

    def test_status_structure(self, api_client):
        data = unwrap(api_client.get("/api/status"))
        for key in ("running", "uptime", "total_capital", "paper_trading"):
            assert key in data
        assert data["running"] is False   # no system running in tests

    def test_metrics_structure(self, api_client):
        data = unwrap(api_client.get("/api/metrics"))
        for key in ("total_pnl", "total_roi", "daily_pnl", "active_trades",
                    "win_rate", "total_trades", "max_drawdown", "daily_trades"):
            assert key in data, f"Missing field: {key}"

    def test_equity_curve_structure(self, api_client):
        data = unwrap(api_client.get("/api/equity"))
        assert "equity_curve" in data
        assert len(data["equity_curve"]) > 0
        point = data["equity_curve"][0]
        assert "timestamp" in point and "equity" in point

    def test_bots_returns_list(self, api_client):
        data = unwrap(api_client.get("/api/bots"))
        assert isinstance(data["bots"], list)
        # Every bot must carry an `id` field (used by toggle/config endpoints)
        for bot in data["bots"]:
            assert "id" in bot

    def test_recent_trades_structure(self, api_client):
        data = unwrap(api_client.get("/api/trades/recent"))
        assert "trades" in data
        if data["trades"]:
            trade = data["trades"][0]
            for key in ("id", "symbol", "direction", "pnl", "status", "time"):
                assert key in trade
            # direction must be lowercase
            assert trade["direction"] in ("long", "short")

    def test_logs_are_structured_objects(self, api_client):
        data = unwrap(api_client.get("/api/logs"))
        assert "logs" in data
        assert len(data["logs"]) > 0
        log = data["logs"][0]
        for key in ("timestamp", "level", "message", "source"):
            assert key in log, f"Log entry must be a structured object, missing {key}"


class TestConfigEndpoints:
    def test_get_config(self, api_client):
        data = unwrap(api_client.get("/api/config"))
        assert "trading_mode" in data
        assert "total_capital" in data

    def test_get_config_full(self, api_client):
        data = unwrap(api_client.get("/api/config/full"))
        assert "global_config" in data
        assert isinstance(data["bot_configs"], list)
        # Each bot config must carry an `id`
        for bot in data["bot_configs"]:
            assert "id" in bot

    def test_post_config(self, api_client):
        payload = {
            "trading_mode": "paper",
            "total_capital": 1500.0,
            "daily_loss_limit": 0.04,
            "daily_profit_target": 0.025,
        }
        data = unwrap(api_client.post("/api/config", json=payload))
        assert "message" in data

    def test_config_backups(self, api_client):
        data = unwrap(api_client.get("/api/config/backups"))
        assert "backups" in data
        assert isinstance(data["backups"], list)


class TestBotManagementEndpoints:
    def test_available_symbols(self, api_client):
        data = unwrap(api_client.get("/api/bots/available-symbols"))
        assert len(data["symbols"]) > 0
        assert all(isinstance(s, str) for s in data["symbols"])

    def test_available_timeframes(self, api_client):
        data = unwrap(api_client.get("/api/bots/available-timeframes"))
        assert "5m" in data["timeframes"]
        assert "1h" in data["timeframes"]

    def test_bot_count(self, api_client):
        data = unwrap(api_client.get("/api/bots/count"))
        # New shape: current_count / maximum_allowed / can_add_more
        assert "current_count" in data
        assert "maximum_allowed" in data
        assert "can_add_more" in data

    def test_get_bot_config_503_without_system(self, api_client):
        r = api_client.get("/api/bots/LINK_USDT_5m/config")
        assert r.status_code == 503

    def test_toggle_bot_503_without_system(self, api_client):
        r = api_client.post("/api/bots/LINK_USDT_5m/toggle", json={"enabled": False})
        assert r.status_code == 503

    def test_add_bot_503_without_system(self, api_client):
        payload = {"symbol": "BTC/USDT", "timeframe": "15m",
                   "capital_allocation": 0.3, "max_risk_per_trade": 0.02}
        r = api_client.post("/api/bots/add", json=payload)
        assert r.status_code == 503

    def test_update_bot_503_without_system(self, api_client):
        r = api_client.put("/api/bots/update/0", json={"enabled": False})
        assert r.status_code == 503

    def test_remove_bot_503_without_system(self, api_client):
        r = api_client.delete("/api/bots/remove/0")
        assert r.status_code == 503


class TestBotManagementWithSystem:
    """Tests that require a live trading system instance."""

    @pytest.fixture(autouse=True)
    def _setup_system(self, api_client):
        import api_server
        from production_trading_system import (
            ProductionTradingSystem, GlobalConfig, BotConfig, OptimizedSignalGenerator)

        global_cfg = GlobalConfig(total_capital=1200.0, paper_trading=True,
                                  max_concurrent_trades=2)
        bot = BotConfig("LINK/USDT", "5m", 0.7, 0.025, 0.65, 0.018, 0.035)
        system = object.__new__(ProductionTradingSystem)
        system.global_config = global_cfg
        system.bot_configs = {"LINK/USDT_5m": bot}
        system.trade_history = []
        system.active_trades = {}
        system.equity_curve = []
        system.total_pnl = 0.0
        system.daily_pnl = 0.0
        system.daily_trades = 0
        system.signal_generators = {"LINK/USDT_5m": OptimizedSignalGenerator("LINK/USDT", "5m")}
        system.risk_managers = {}

        api_server.trading_system = system
        yield
        api_server.trading_system = None

    def test_bots_returns_real_data(self, api_client):
        data = unwrap(api_client.get("/api/bots"))
        assert len(data["bots"]) == 1
        assert data["bots"][0]["symbol"] == "LINK/USDT"
        assert data["bots"][0]["id"] == "LINK/USDT_5m"

    def test_get_bot_config(self, api_client):
        # bot_id contains "/" — route uses {bot_id:path}
        data = unwrap(api_client.get("/api/bots/LINK/USDT_5m/config"))
        assert data["symbol"] == "LINK/USDT"
        assert data["id"] == "LINK/USDT_5m"

    def test_toggle_bot(self, api_client):
        data = unwrap(api_client.post("/api/bots/LINK/USDT_5m/toggle", json={"enabled": False}))
        assert data["enabled"] is False

    def test_add_and_remove_bot(self, api_client):
        payload = {"symbol": "BTC/USDT", "timeframe": "15m",
                   "capital_allocation": 0.3, "max_risk_per_trade": 0.02}
        unwrap(api_client.post("/api/bots/add", json=payload))

        count = unwrap(api_client.get("/api/bots/count"))
        assert count["current_count"] == 2

        unwrap(api_client.delete("/api/bots/remove/1"))

        count2 = unwrap(api_client.get("/api/bots/count"))
        assert count2["current_count"] == 1

    def test_add_duplicate_bot_returns_409(self, api_client):
        payload = {"symbol": "LINK/USDT", "timeframe": "5m",
                   "capital_allocation": 0.3, "max_risk_per_trade": 0.02}
        r = api_client.post("/api/bots/add", json=payload)
        assert r.status_code == 409

    def test_update_bot_by_id(self, api_client):
        data = unwrap(api_client.put("/api/config/bot/LINK/USDT_5m",
                                     json={"enabled": False}))
        assert "message" in data


class TestAnalyticsEndpoints:
    def test_performance_metrics_system(self, api_client):
        """New /performance-metrics returns CPU/memory, not trading metrics."""
        data = unwrap(api_client.get("/api/performance-metrics"))
        assert "system_metrics" in data
        assert "application_metrics" in data
        assert "trading_metrics" in data
        assert "memory_percent" in data["system_metrics"]
        assert "system_running" in data["trading_metrics"]

    def test_daily_stats(self, api_client):
        data = unwrap(api_client.get("/api/daily-stats"))
        assert "daily_stats" in data

    def test_signal_quality(self, api_client):
        data = unwrap(api_client.get("/api/signal-quality"))
        for key in ("current_quality_score", "pass_rate", "layer_scores", "recent_rejections"):
            assert key in data
        for layer in ("technical", "market_structure", "binance_sentiment", "ml_confidence"):
            assert layer in data["layer_scores"]

    def test_market_regime(self, api_client):
        data = unwrap(api_client.get("/api/market-regime/BTC/USDT"))
        assert "regime" in data
        assert "strategy_config" in data
        assert "strategy_type" in data["strategy_config"]

    def test_market_sentiment(self, api_client):
        data = unwrap(api_client.get("/api/market-sentiment/BTC/USDT"))
        assert "sentiment_label" in data
        assert "raw_data" in data
        assert "long_short_ratio" in data["raw_data"]

    def test_logs_statistics(self, api_client):
        data = unwrap(api_client.get("/api/logs/statistics"))
        assert "total_entries" in data
        assert "error_count" in data

    def test_logs_categories(self, api_client):
        data = unwrap(api_client.get("/api/logs/categories"))
        assert isinstance(data["categories"], list)


class TestPortfolioEndpoints:
    def test_portfolio_status_when_idle(self, api_client):
        data = unwrap(api_client.get("/api/portfolio"))
        assert data["running"] is False
        assert data["positions"] == []
        assert "total_pnl" in data

    def test_portfolio_stop_when_idle_400(self, api_client):
        r = api_client.post("/api/portfolio/stop")
        assert r.status_code == 400


class TestCollectorEndpoints:
    def test_collector_status_idle(self, api_client):
        data = unwrap(api_client.get("/api/collector"))
        assert data["running"] is False

    def test_collector_stop_idle_400(self, api_client):
        r = api_client.post("/api/collector/stop")
        assert r.status_code == 400


class TestBacktestEndpoint:
    def test_backtest_trigger(self, api_client):
        data = unwrap(api_client.post("/api/backtest"))
        assert "message" in data

    def test_backtest_results_endpoint(self, api_client):
        # Returns {success, data} whether or not results exist
        unwrap(api_client.get("/api/backtest/results"))
