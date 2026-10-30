"""Pure-function tests for backend.agent.guardrails — no network, no mocks."""
import pytest

from backend.agent.guardrails import (
    SYMBOL_ALLOWLIST,
    daily_loss_tripped,
    daily_trade_cap_exceeded,
    drawdown_tripped,
    llm_call_budget_exceeded,
    validate_order_request,
)


class TestValidateOrderRequest:
    def test_well_formed_order_accepted(self):
        ok, reason = validate_order_request(
            {"symbol": "BTC/USDT", "side": "buy", "notional_usd": 50.0},
            equity=1000.0, open_notional={},
        )
        assert ok is True
        assert reason == "ok"

    def test_symbol_outside_allowlist_rejected(self):
        ok, reason = validate_order_request(
            {"symbol": "DOGE/USDT", "side": "buy", "notional_usd": 50.0},
            equity=1000.0, open_notional={},
        )
        assert ok is False
        assert "allow-list" in reason

    def test_hallucinated_symbol_rejected(self):
        """A plausible-looking but nonexistent symbol must still be rejected."""
        ok, reason = validate_order_request(
            {"symbol": "BTCUSDT", "side": "buy", "notional_usd": 50.0},  # missing '/'
            equity=1000.0, open_notional={},
        )
        assert ok is False

    def test_invalid_side_rejected(self):
        ok, reason = validate_order_request(
            {"symbol": "BTC/USDT", "side": "long", "notional_usd": 50.0},
            equity=1000.0, open_notional={},
        )
        assert ok is False
        assert "side" in reason

    def test_oversized_notional_rejected_not_clamped(self):
        ok, reason = validate_order_request(
            {"symbol": "BTC/USDT", "side": "buy", "notional_usd": 500.0},
            equity=1000.0, open_notional={}, max_position_pct=0.15,
        )
        assert ok is False
        assert "max_position_pct" in reason

    def test_non_numeric_notional_rejected(self):
        ok, reason = validate_order_request(
            {"symbol": "BTC/USDT", "side": "buy", "notional_usd": "a lot"},
            equity=1000.0, open_notional={},
        )
        assert ok is False

    def test_bool_notional_rejected(self):
        """bool is technically an int subclass in Python — must not sneak past
        the numeric check."""
        ok, reason = validate_order_request(
            {"symbol": "BTC/USDT", "side": "buy", "notional_usd": True},
            equity=1000.0, open_notional={},
        )
        assert ok is False

    def test_negative_notional_rejected(self):
        ok, reason = validate_order_request(
            {"symbol": "BTC/USDT", "side": "buy", "notional_usd": -10.0},
            equity=1000.0, open_notional={},
        )
        assert ok is False

    def test_max_open_positions_blocks_new_symbol(self):
        open_notional = {"BTC/USDT": 50.0, "ETH/USDT": 50.0, "SOL/USDT": 50.0}
        ok, reason = validate_order_request(
            {"symbol": "BTC/USDT", "side": "buy", "notional_usd": 10.0},
            equity=1000.0, open_notional=open_notional, max_open_positions=3,
        )
        # BTC/USDT is already open — adding to it is fine (not a new position)
        assert ok is True

    def test_max_open_positions_blocks_truly_new_symbol(self):
        open_notional = {s: 50.0 for s in SYMBOL_ALLOWLIST}
        # all 3 symbols already open at the cap — this exact test only makes
        # sense if there's a 4th candidate, so assert the cap logic directly
        # via a smaller allow-list scenario instead:
        ok, reason = validate_order_request(
            {"symbol": "ETH/USDT", "side": "buy", "notional_usd": 10.0},
            equity=1000.0, open_notional={"BTC/USDT": 50.0, "SOL/USDT": 50.0},
            max_open_positions=2,
        )
        assert ok is False
        assert "max_open_positions" in reason

    def test_malformed_payload_type_rejected(self):
        ok, reason = validate_order_request("not a dict", equity=1000.0, open_notional={})
        assert ok is False

    def test_zero_equity_rejected(self):
        ok, reason = validate_order_request(
            {"symbol": "BTC/USDT", "side": "buy", "notional_usd": 10.0},
            equity=0.0, open_notional={},
        )
        assert ok is False


class TestKillSwitches:
    def test_daily_loss_not_tripped_below_limit(self):
        assert daily_loss_tripped(daily_pnl=-10.0, start_equity=1000.0, limit_pct=0.05) is False

    def test_daily_loss_tripped_at_limit(self):
        assert daily_loss_tripped(daily_pnl=-50.0, start_equity=1000.0, limit_pct=0.05) is True

    def test_daily_loss_ignores_gains(self):
        assert daily_loss_tripped(daily_pnl=100.0, start_equity=1000.0, limit_pct=0.05) is False

    def test_drawdown_not_tripped(self):
        assert drawdown_tripped(peak_equity=1000.0, equity=900.0, limit_pct=0.20) is False

    def test_drawdown_tripped_at_limit(self):
        assert drawdown_tripped(peak_equity=1000.0, equity=800.0, limit_pct=0.20) is True

    def test_drawdown_zero_peak_safe(self):
        assert drawdown_tripped(peak_equity=0.0, equity=0.0, limit_pct=0.20) is False

    def test_daily_trade_cap(self):
        assert daily_trade_cap_exceeded(9, cap=10) is False
        assert daily_trade_cap_exceeded(10, cap=10) is True

    def test_llm_call_budget(self):
        assert llm_call_budget_exceeded(149, cap=150) is False
        assert llm_call_budget_exceeded(150, cap=150) is True
