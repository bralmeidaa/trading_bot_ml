"""
API endpoint tests — verify every route defined in api_server.py works correctly,
especially the 11 endpoints added for the React frontend.

Tests run against a TestClient with no live trading system (demo data mode).
"""
import pytest
from fastapi.testclient import TestClient


class TestCoreEndpoints:
    def test_health_check(self, api_client):
        r = api_client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_status_structure(self, api_client):
        r = api_client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert "running" in data
        assert "uptime" in data
        assert "total_capital" in data
        assert "paper_trading" in data
        assert data["running"] is False   # no system running in tests

    def test_metrics_structure(self, api_client):
        r = api_client.get("/api/metrics")
        assert r.status_code == 200
        data = r.json()
        for key in ("total_pnl", "total_roi", "daily_pnl", "active_trades",
                    "win_rate", "total_trades", "max_drawdown"):
            assert key in data, f"Missing field: {key}"

    def test_equity_curve_structure(self, api_client):
        r = api_client.get("/api/equity")
        assert r.status_code == 200
        data = r.json()
        assert "equity_curve" in data
        assert len(data["equity_curve"]) > 0
        point = data["equity_curve"][0]
        assert "timestamp" in point and "equity" in point

    def test_bots_returns_list(self, api_client):
        r = api_client.get("/api/bots")
        assert r.status_code == 200
        data = r.json()
        assert "bots" in data
        assert isinstance(data["bots"], list)

    def test_recent_trades_structure(self, api_client):
        r = api_client.get("/api/trades/recent")
        assert r.status_code == 200
        data = r.json()
        assert "trades" in data
        if data["trades"]:
            trade = data["trades"][0]
            for key in ("symbol", "direction", "pnl", "status", "time"):
                assert key in trade

    def test_logs_structure(self, api_client):
        r = api_client.get("/api/logs")
        assert r.status_code == 200
        assert "logs" in r.json()


class TestConfigEndpoints:
    def test_get_config(self, api_client):
        r = api_client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert "trading_mode" in data
        assert "total_capital" in data

    def test_get_config_full(self, api_client):
        r = api_client.get("/api/config/full")
        assert r.status_code == 200
        data = r.json()
        assert "global_config" in data
        assert "bot_configs" in data
        assert isinstance(data["bot_configs"], list)

    def test_post_config(self, api_client):
        payload = {
            "trading_mode": "paper",
            "total_capital": 1500.0,
            "daily_loss_limit": 0.04,
            "daily_profit_target": 0.025,
        }
        r = api_client.post("/api/config", json=payload)
        assert r.status_code == 200
        assert "message" in r.json()


class TestBotManagementEndpoints:
    def test_available_symbols(self, api_client):
        r = api_client.get("/api/bots/available-symbols")
        assert r.status_code == 200
        data = r.json()
        assert "symbols" in data
        assert len(data["symbols"]) > 0
        assert all(isinstance(s, str) for s in data["symbols"])

    def test_available_timeframes(self, api_client):
        r = api_client.get("/api/bots/available-timeframes")
        assert r.status_code == 200
        data = r.json()
        assert "timeframes" in data
        assert "5m" in data["timeframes"]
        assert "1h" in data["timeframes"]

    def test_bot_count(self, api_client):
        r = api_client.get("/api/bots/count")
        assert r.status_code == 200
        assert "count" in r.json()
        assert isinstance(r.json()["count"], int)

    def test_get_bot_config_503_without_system(self, api_client):
        """Without a running trading system, must return 503."""
        r = api_client.get("/api/bots/LINK_USDT_5m/config")
        assert r.status_code == 503

    def test_toggle_bot_503_without_system(self, api_client):
        r = api_client.post("/api/bots/LINK_USDT_5m/toggle", json={"enabled": False})
        assert r.status_code == 503

    def test_add_bot_503_without_system(self, api_client):
        payload = {
            "symbol": "BTC/USDT", "timeframe": "15m",
            "capital_allocation": 0.3, "max_risk_per_trade": 0.02,
        }
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
        """Inject a live trading system into the api_server module."""
        import api_server
        from production_trading_system import ProductionTradingSystem, GlobalConfig, BotConfig

        global_cfg = GlobalConfig(total_capital=1200.0, paper_trading=True,
                                   max_concurrent_trades=2)
        bot_cfgs = [
            BotConfig("LINK/USDT", "5m", 0.7, 0.025, 0.65, 0.018, 0.035),
        ]
        system = object.__new__(ProductionTradingSystem)
        system.global_config = global_cfg
        system.bot_configs = {"LINK/USDT_5m": bot_cfgs[0]}
        system.trade_history = []
        system.active_trades = {}
        system.equity_curve = []
        system.total_pnl = 0.0
        system.daily_pnl = 0.0
        system.signal_generators = {}
        system.risk_managers = {}
        from production_trading_system import OptimizedSignalGenerator
        system.signal_generators["LINK/USDT_5m"] = OptimizedSignalGenerator("LINK/USDT", "5m")

        api_server.trading_system = system
        yield
        api_server.trading_system = None

    def test_bots_returns_real_data(self, api_client):
        r = api_client.get("/api/bots")
        assert r.status_code == 200
        bots = r.json()["bots"]
        assert len(bots) == 1
        assert bots[0]["symbol"] == "LINK/USDT"
        assert bots[0]["timeframe"] == "5m"

    def test_get_bot_config(self, api_client):
        # bot_id contains "/" — route uses {bot_id:path} to handle this
        r = api_client.get("/api/bots/LINK/USDT_5m/config")
        assert r.status_code == 200
        assert r.json()["symbol"] == "LINK/USDT"

    def test_toggle_bot(self, api_client):
        r = api_client.post("/api/bots/LINK/USDT_5m/toggle", json={"enabled": False})
        assert r.status_code == 200
        assert r.json()["enabled"] is False

    def test_add_and_remove_bot(self, api_client):
        payload = {
            "symbol": "BTC/USDT", "timeframe": "15m",
            "capital_allocation": 0.3, "max_risk_per_trade": 0.02,
        }
        r_add = api_client.post("/api/bots/add", json=payload)
        assert r_add.status_code == 200

        r_count = api_client.get("/api/bots/count")
        assert r_count.json()["count"] == 2

        r_remove = api_client.delete("/api/bots/remove/1")
        assert r_remove.status_code == 200

        r_count2 = api_client.get("/api/bots/count")
        assert r_count2.json()["count"] == 1

    def test_add_duplicate_bot_returns_409(self, api_client):
        payload = {
            "symbol": "LINK/USDT", "timeframe": "5m",
            "capital_allocation": 0.3, "max_risk_per_trade": 0.02,
        }
        r = api_client.post("/api/bots/add", json=payload)
        assert r.status_code == 409


class TestPerformanceMetrics:
    def test_performance_metrics_extended(self, api_client):
        r = api_client.get("/api/performance-metrics")
        assert r.status_code == 200
        data = r.json()
        assert "sharpe_ratio" in data
        assert "profit_factor" in data
        assert "total_pnl" in data

    def test_daily_stats(self, api_client):
        r = api_client.get("/api/daily-stats")
        assert r.status_code == 200
        assert "daily_stats" in r.json()


class TestBacktestEndpoint:
    def test_backtest_trigger(self, api_client):
        r = api_client.post("/api/backtest")
        assert r.status_code == 200
        assert "message" in r.json()

    def test_backtest_results_endpoint(self, api_client):
        r = api_client.get("/api/backtest/results")
        assert r.status_code == 200  # returns message if no results yet
