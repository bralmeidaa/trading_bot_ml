"""
Daily AI agent email digest — monitoring for the non-deterministic paper
trading agent (backend/agent/). The decision log (AgentDecisionLog) is the
system of record; this just turns a day's worth of it into something a
human reads once a day instead of curling /api/agent/decisions by hand.

Design mirrors backend/alerts/regime_alert.py: pure functions (build_digest,
compose_message) are network-free and unit-tested; run_once() wires SMTP
(reusing regime_alert's smtp_config_from_env/send_email — not duplicated).
Escalates (different subject) if the drawdown kill-switch tripped today —
that's the one thing that must not go unnoticed for a day.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional

from backend.alerts.regime_alert import send_email, smtp_config_from_env

logger = logging.getLogger(__name__)


def _parse_json_field(raw: Optional[str], default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def build_digest(decisions: List[dict], status: dict, today: Optional[str] = None) -> dict:
    """
    Pure: summarize today's decision-log entries + current agent status.
    `decisions` is whatever AgentDecisionLogRepository.get_recent() returns
    (newest first); `today` is an ISO date string, defaults to UTC today.
    """
    today = today or datetime.utcnow().date().isoformat()
    todays = [d for d in decisions if (d.get("timestamp") or "")[:10] == today]

    trades, holds, rejections, halts = [], 0, [], []
    for d in todays:
        executed = _parse_json_field(d.get("tool_calls_executed"), [])
        rejected = _parse_json_field(d.get("guardrail_rejections"), [])
        if any(c.get("name") == "place_paper_order" and c.get("output", {}).get("executed")
              for c in executed):
            trades.append(d)
        elif d.get("halted_reason"):
            halts.append(d)
        else:
            holds += 1
        if rejected:
            rejections.append(d)

    kill_switch_hit = any(d.get("halted_reason") == "drawdown_kill_switch" for d in halts)
    llm_calls = sum(1 for d in todays if not d.get("halted_reason"))  # halted cycles skip the LLM
    tokens = sum((d.get("prompt_tokens") or 0) + (d.get("completion_tokens") or 0) for d in todays)

    return {
        "date": today,
        "cycles_today": len(todays),
        "trades": trades,
        "holds": holds,
        "rejections": rejections,
        "halts": halts,
        "kill_switch_hit": kill_switch_hit,
        "llm_calls": llm_calls,
        "tokens": tokens,
        "equity": status.get("equity", 0.0),
        "total_pnl": status.get("total_pnl", 0.0),
        "daily_pnl": status.get("daily_pnl", 0.0),
        "positions": status.get("positions", []),
        "running": status.get("running", False),
        "halted": status.get("halted", False),
    }


def compose_message(digest: dict) -> tuple[str, str]:
    """Build (subject, body) from a digest. Escalates on kill-switch."""
    if digest["kill_switch_hit"]:
        subject = "🛑 Agente de IA: KILL-SWITCH disparado (drawdown)"
        head = ("O drawdown passou do limite configurado hoje — o agente fechou "
                "todas as posições e PAROU de operar. Precisa de atenção/reset manual.")
    elif not digest["running"]:
        subject = "⚠️ Agente de IA: não está rodando"
        head = "O motor do agente não está ativo no momento (parado ou nunca iniciou)."
    else:
        subject = f"Agente de IA — resumo do dia ({digest['date']}): {len(digest['trades'])} trade(s)"
        head = "Resumo diário de operação (paper trading)."

    lines = [
        head, "",
        f"Ciclos hoje: {digest['cycles_today']}  |  Trades: {len(digest['trades'])}  |  "
        f"Holds: {digest['holds']}  |  Rejeições de guardrail: {len(digest['rejections'])}",
        f"Equity: ${digest['equity']:,.2f}  |  PnL total: ${digest['total_pnl']:+,.2f}  |  "
        f"PnL hoje: ${digest['daily_pnl']:+,.2f}",
        f"Chamadas LLM hoje: {digest['llm_calls']}  |  Tokens: {digest['tokens']:,}",
    ]

    if digest["positions"]:
        lines += ["", "Posições abertas:"]
        for p in digest["positions"]:
            lines.append(f"  {p.get('symbol')}: {p.get('side')} ${p.get('notional', 0):,.2f} "
                        f"@ {p.get('entry_price')}")

    if digest["trades"]:
        lines += ["", "Trades executados hoje:"]
        for d in digest["trades"]:
            lines.append(f"  [{d.get('timestamp', '')[11:16]}] {(d.get('rationale') or '')[:200]}")

    if digest["rejections"]:
        lines += ["", "Ordens rejeitadas pelo guardrail hoje (o agente tentou e foi bloqueado):"]
        for d in digest["rejections"]:
            rej = _parse_json_field(d.get("guardrail_rejections"), [])
            for r in rej:
                lines.append(f"  [{d.get('timestamp', '')[11:16]}] {r.get('output', {}).get('rejected_reason', '')}")

    if digest["halts"] and not digest["kill_switch_hit"]:
        lines += ["", "Ciclos parados sem chamar a LLM hoje (orçamento/limite diário, não é erro):"]
        for d in digest["halts"]:
            lines.append(f"  [{d.get('timestamp', '')[11:16]}] {d.get('halted_reason')}")

    return subject, "\n".join(lines)


def run_once(decisions: List[dict], status: dict, smtp_cfg: Optional[dict] = None,
            force: bool = False) -> dict:
    """Compose + send (if SMTP configured). `force=True` sends even with a
    quiet day (used by the manual test endpoint)."""
    digest = build_digest(decisions, status)
    subject, body = compose_message(digest)
    if force:
        subject = "[teste] " + subject

    cfg = smtp_cfg or smtp_config_from_env()
    sent = False
    if cfg:
        try:
            send_email(cfg, subject, body)
            sent = True
        except Exception as exc:
            logger.error(f"agent alert: email send failed: {exc}")
    else:
        logger.info(f"agent alert (no SMTP configured): {subject}\n{body}")

    return {"digest": digest, "sent": sent, "smtp_configured": cfg is not None, "subject": subject}
