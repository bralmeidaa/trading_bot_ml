"""
Daily BTC-regime alert (email).

Sends a once-a-day heartbeat with the current regime state (ON/OFF, BTC EMA gap,
portfolio equity/positions) AND highlights the day the regime FLIPS — the moment
the cross-sectional strategy switches between operating and capital-preservation.

Design: pure functions (regime_snapshot, detect_flip, compose_message) are
network-free and unit-tested; run_once() wires I/O (panel fetch, SMTP, state file).
State is persisted on the durable host mount so flip detection survives redeploys.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from typing import Optional

import pandas as pd

from backend.strategy.cross_sectional import btc_regime

logger = logging.getLogger(__name__)

# State on the persistent mount (same place as the DB / order book) so a redeploy
# doesn't reset the last-known regime and spuriously re-fire a flip alert.
STATE_PATH = os.getenv("REGIME_STATE_PATH", "/app/data/regime_state.json")


def regime_snapshot(close: pd.DataFrame, lookback: int = 12) -> dict:
    """Current regime state from a daily close panel (must include BTC/USDT)."""
    reg = btc_regime(close, lookback)
    btc = close["BTC/USDT"]
    e_fast = btc.ewm(span=max(2, lookback)).mean().iloc[-1]
    e_slow = btc.ewm(span=max(4, lookback * 5)).mean().iloc[-1]
    return {
        "regime": "ON" if bool(reg.iloc[-1]) else "OFF",
        "btc_price": float(btc.iloc[-1]),
        "ema_fast": float(e_fast),
        "ema_slow": float(e_slow),
        "gap_pct": float(e_fast / e_slow - 1.0) * 100.0,
        "date": pd.to_datetime(close.index[-1], unit="ms").date().isoformat(),
    }


def detect_flip(prev_regime: Optional[str], cur_regime: str) -> Optional[str]:
    """Return 'OFF->ON' / 'ON->OFF' if the regime changed vs the stored state."""
    if prev_regime and prev_regime != cur_regime:
        return f"{prev_regime}->{cur_regime}"
    return None


def compose_message(snap: dict, portfolio: Optional[dict], flip: Optional[str]) -> tuple[str, str]:
    """Build (subject, body). Highlights a flip; otherwise a plain heartbeat."""
    if flip == "OFF->ON":
        subject = "🟢 BTC regime VIROU ON — a estratégia vai operar"
        head = ("O regime do BTC cruzou para ON. No próximo rebalance a carteira "
                "monta as pernas long/short (sai do flat).")
    elif flip == "ON->OFF":
        subject = "🔴 BTC regime VIROU OFF — a estratégia vai para flat"
        head = ("O regime do BTC cruzou para OFF. A carteira zera posições no "
                "próximo rebalance (preservação de capital).")
    else:
        subject = f"Regime BTC: {snap['regime']} (gap {snap['gap_pct']:+.1f}%)"
        head = ("Sem virada hoje — heartbeat diário.")

    lines = [
        head, "",
        f"Regime: {snap['regime']}",
        f"BTC: ${snap['btc_price']:,.0f}  |  EMA rápida ${snap['ema_fast']:,.0f}  "
        f"|  EMA lenta ${snap['ema_slow']:,.0f}  |  gap {snap['gap_pct']:+.1f}%",
        f"(ON quando a EMA rápida cruza acima da lenta; falta {-snap['gap_pct']:+.1f}% "
        f"pra virar)" if snap["regime"] == "OFF" else
        f"(margem de {snap['gap_pct']:+.1f}% acima do cruzamento)",
    ]
    if portfolio:
        lines += [
            "",
            f"Carteira (paper): equity ${portfolio.get('equity', 0):,.2f}  |  "
            f"posições {len(portfolio.get('positions', []))}  |  "
            f"rebalances {portfolio.get('rebalances', 0)}  |  "
            f"exposure {portfolio.get('config', {}).get('exposure', '?')}",
        ]
    lines += ["", f"Data do sinal: {snap['date']}"]
    return subject, "\n".join(lines)


def _env(name: str) -> Optional[str]:
    """Env var, treating empty or un-expanded Azure placeholders ($(VAR)) as unset."""
    v = os.getenv(name)
    if not v or v.startswith("$("):
        return None
    return v


def smtp_config_from_env() -> Optional[dict]:
    """Read SMTP config from env; None if not configured (→ alert no-ops)."""
    host, user, pw = _env("SMTP_HOST"), _env("SMTP_USER"), _env("SMTP_PASS")
    to = _env("ALERT_TO") or user
    if not (host and user and pw and to):
        return None
    return {"host": host, "port": int(_env("SMTP_PORT") or "587"),
            "user": user, "password": pw, "to": to}


def send_email(cfg: dict, subject: str, body: str) -> None:
    """Send a plain-text email via SMTP+STARTTLS (Gmail/Outlook app password)."""
    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = cfg["user"]
    msg["To"] = cfg["to"]
    with smtplib.SMTP(cfg["host"], cfg["port"], timeout=30) as s:
        s.starttls()
        s.login(cfg["user"], cfg["password"])
        s.sendmail(cfg["user"], [cfg["to"]], msg.as_string())


def _read_state() -> Optional[str]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("regime")
    except Exception:
        return None


def _write_state(snap: dict) -> None:
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"regime": snap["regime"], "date": snap["date"]}, f)
    except Exception as exc:
        logger.warning(f"regime alert: could not persist state: {exc}")


def run_once(close: pd.DataFrame, portfolio: Optional[dict] = None,
             lookback: int = 12, smtp_cfg: Optional[dict] = None,
             force: bool = False) -> dict:
    """
    Compute the snapshot, detect a flip vs stored state, send the email (if SMTP
    configured), persist the new state. `force=True` sends even with no flip
    (used by the manual test endpoint). Returns a small result dict.
    """
    snap = regime_snapshot(close, lookback)
    prev = _read_state()
    flip = detect_flip(prev, snap["regime"])
    subject, body = compose_message(snap, portfolio, flip)

    if force:
        subject = "[teste] " + subject
    cfg = smtp_cfg or smtp_config_from_env()
    sent = False
    if cfg:                            # daily heartbeat → always send when configured
        try:
            send_email(cfg, subject, body)
            sent = True
        except Exception as exc:
            logger.error(f"regime alert: email send failed: {exc}")
    else:
        logger.info(f"regime alert (no SMTP configured): {subject}\n{body}")

    _write_state(snap)
    return {"snapshot": snap, "flip": flip, "sent": sent,
            "smtp_configured": cfg is not None, "subject": subject}
