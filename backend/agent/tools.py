"""
The 5 tools the AI agent can call — deliberately minimal (one agent, a few
tools, not a multi-agent system). Each tool has a JSON-schema (for the
`tools=[...]` chat-completions parameter) and a plain Python implementation.

Funding/open-interest are NOT exposed as tools in v1: this project's own
research already found they carry no numeric predictive lift on their own —
not worth the extra fetch latency/cost for a first version. get_market_context
stays to simple price/volume context; the differentiator here is the news
tool, not more numeric features.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List

from backend.agent.guardrails import SYMBOL_ALLOWLIST
from backend.agent.news_sources import digest_for_symbols
from backend.agent.paper_execution import PaperBroker

logger = logging.getLogger(__name__)


TOOL_SCHEMAS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_market_context",
            "description": ("Get recent price action for the tradeable symbols: "
                            "last close and % change over 1h/24h/7d."),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {"type": "array", "items": {"type": "string"},
                               "description": f"Subset of {SYMBOL_ALLOWLIST}. Omit for all."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_digest",
            "description": ("Get recent news/announcements (last 24h by default) "
                            "relevant to the tradeable symbols or broad crypto market."),
            "parameters": {
                "type": "object",
                "properties": {
                    "hours": {"type": "integer", "description": "Lookback window in hours.", "default": 24},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_positions",
            "description": "Get currently open paper positions and account equity.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_paper_order",
            "description": ("Open or add to a paper position. Will be rejected by "
                            "risk guardrails if it violates position/size limits — "
                            "check the rejection reason if it fails."),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "enum": SYMBOL_ALLOWLIST},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "notional_usd": {"type": "number", "description": "USD size of the position."},
                    "rationale": {"type": "string",
                                 "description": "REQUIRED: why this trade, citing the specific "
                                                "price/news context that motivated it."},
                },
                "required": ["symbol", "side", "notional_usd", "rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hold",
            "description": "Explicitly decide to do nothing this cycle. Always valid.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rationale": {"type": "string", "description": "Why no trade this cycle."},
                },
                "required": ["rationale"],
            },
        },
    },
]


def build_tool_dispatch(universe: List[str], exchange, broker: PaperBroker,
                        guardrail_check: Callable[[dict], tuple]) -> Dict[str, Callable[[dict], dict]]:
    """
    Build the name->handler map passed to AgentLLMClient.chat_with_tools.
    `guardrail_check(payload) -> (ok: bool, reason: str)` is injected so this
    module doesn't need to know the engine's current equity/open-notional —
    the engine closes over that state when constructing the check.
    """

    def _get_market_context(args: dict) -> dict:
        symbols = args.get("symbols") or universe
        symbols = [s for s in symbols if s in universe]
        out = {}
        for sym in symbols:
            try:
                ticker = exchange.fetch_ticker(sym)
                out[sym] = {
                    "last": ticker.get("last"),
                    "change_24h_pct": ticker.get("percentage"),
                    "high_24h": ticker.get("high"),
                    "low_24h": ticker.get("low"),
                    "volume_24h": ticker.get("quoteVolume"),
                }
            except Exception as exc:
                out[sym] = {"error": str(exc)}
        return {"prices": out}

    def _get_news_digest(args: dict) -> dict:
        hours = args.get("hours", 24)
        return {"news": digest_for_symbols(universe, hours=hours)}

    def _get_positions(args: dict) -> dict:
        return {"equity": broker.equity, "positions": broker.positions()}

    def _place_paper_order(args: dict) -> dict:
        ok, reason = guardrail_check(args)
        if not ok:
            return {"executed": False, "rejected_reason": reason}
        trade = broker.place_order(
            symbol=args["symbol"], side=args["side"],
            notional_usd=float(args["notional_usd"]), rationale=args.get("rationale", ""),
        )
        return {"executed": True, "trade_id": trade.id, "entry_price": trade.entry_price}

    def _hold(args: dict) -> dict:
        return {"executed": True, "action": "hold", "rationale": args.get("rationale", "")}

    return {
        "get_market_context": _get_market_context,
        "get_news_digest": _get_news_digest,
        "get_positions": _get_positions,
        "place_paper_order": _place_paper_order,
        "hold": _hold,
    }
