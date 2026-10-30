"""Tests for backend/alerts/regime_alert.py pure functions (no network/SMTP)."""
import numpy as np
import pandas as pd
import pytest

from backend.alerts.regime_alert import (
    regime_snapshot, detect_flip, compose_message, smtp_config_from_env,
)


def _btc_panel(trend: str, n=120):
    """Daily panel with BTC in an up or down trend to force regime ON/OFF."""
    idx = np.arange(n) * 86400000
    if trend == "up":
        btc = np.linspace(50000, 70000, n)
    else:
        btc = np.linspace(70000, 50000, n)
    return pd.DataFrame({"BTC/USDT": btc, "ETH/USDT": btc * 0.05}, index=idx)


class TestRegimeSnapshot:
    def test_uptrend_is_on(self):
        snap = regime_snapshot(_btc_panel("up"), lookback=12)
        assert snap["regime"] == "ON"
        assert snap["gap_pct"] > 0                       # fast EMA above slow
        assert snap["date"]

    def test_downtrend_is_off(self):
        snap = regime_snapshot(_btc_panel("down"), lookback=12)
        assert snap["regime"] == "OFF"
        assert snap["gap_pct"] < 0


class TestDetectFlip:
    def test_no_prev_no_flip(self):
        assert detect_flip(None, "OFF") is None

    def test_same_no_flip(self):
        assert detect_flip("ON", "ON") is None

    def test_off_to_on(self):
        assert detect_flip("OFF", "ON") == "OFF->ON"

    def test_on_to_off(self):
        assert detect_flip("ON", "OFF") == "ON->OFF"


class TestComposeMessage:
    def _snap(self, regime="OFF"):
        return {"regime": regime, "btc_price": 64000.0, "ema_fast": 63000.0,
                "ema_slow": 66000.0, "gap_pct": -4.8, "date": "2026-07-12"}

    def test_flip_on_subject_highlights(self):
        subj, body = compose_message(self._snap("ON"), None, "OFF->ON")
        assert "ON" in subj and "🟢" in subj
        assert "long/short" in body.lower()

    def test_heartbeat_subject_has_state(self):
        subj, body = compose_message(self._snap("OFF"), None, None)
        assert "OFF" in subj and "heartbeat" in body.lower()

    def test_portfolio_block_included(self):
        pf = {"equity": 1200.0, "positions": [], "rebalances": 3,
              "config": {"exposure": 0.35}}
        _, body = compose_message(self._snap(), pf, None)
        assert "1,200" in body and "0.35" in body


class TestSmtpConfig:
    def test_none_when_unset(self, monkeypatch):
        for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS", "ALERT_TO"):
            monkeypatch.delenv(k, raising=False)
        assert smtp_config_from_env() is None

    def test_built_when_set(self, monkeypatch):
        monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
        monkeypatch.setenv("SMTP_USER", "me@gmail.com")
        monkeypatch.setenv("SMTP_PASS", "app-pw")
        monkeypatch.delenv("ALERT_TO", raising=False)
        cfg = smtp_config_from_env()
        assert cfg["host"] == "smtp.gmail.com" and cfg["port"] == 587
        assert cfg["to"] == "me@gmail.com"        # defaults to user
