"""
The 6 tools the AI agent can call — deliberately minimal (one agent, a few
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
from typing import Callable, Dict, List, Optional

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
            "description": ("Open a paper position, or add to an existing one on the SAME side "
                            "(buy = long, sell = short). Never use it to exit or flip — an order "
                            "against an open position is rejected; use close_paper_position. "
                            "Every position gets an automatic stop-loss and take-profit that are "
                            "enforced every ~30s without you. Rejected by risk guardrails if it "
                            "violates size/risk/exposure limits — read the rejection reason."),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "enum": SYMBOL_ALLOWLIST},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "notional_usd": {"type": "number", "description": "USD size to add."},
                    "stop_pct": {"type": "number",
                                 "description": ("Optional stop distance as a fraction of entry "
                                                 "(0.02 = 2%). Omit to use the ATR-based default.")},
                    "target_pct": {"type": "number",
                                   "description": ("Optional take-profit distance as a fraction of "
                                                   "entry, at least stop_pct. Omit for the default.")},
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
            "name": "close_paper_position",
            "description": ("Close the whole open position on a symbol at the current price "
                            "(realizes its PnL). Use it to take profit early, cut a loss before "
                            "the stop, or before trading the other side."),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "enum": SYMBOL_ALLOWLIST},
                    "rationale": {"type": "string",
                                  "description": "REQUIRED: why exit now, citing current price/news."},
                },
                "required": ["symbol", "rationale"],
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
                        guardrail_check: Callable[[dict], tuple],
                        news_tracker: Optional[Callable[[List[dict]], List[dict]]] = None,
                        order_preparer: Optional[Callable[[dict], dict]] = None,
                        ) -> Dict[str, Callable[[dict], dict]]:
    """
    Build the name->handler map passed to AgentLLMClient.chat_with_tools.
    `guardrail_check(payload) -> (ok: bool, reason: str)` is injected so this
    module doesn't need to know the engine's current equity/open-notional —
    the engine closes over that state when constructing the check.
    `news_tracker(items) -> items`, if given, annotates each news item with
    how many cycles ago it was first seen (so the model can tell a fresh
    headline from one it already reacted to).
    `order_preparer(args) -> args`, if given, fills the default stop/target
    before the guardrail check, so the risk check sees the real bracket.
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
        items = digest_for_symbols(universe, hours=hours)
        if news_tracker:
            items = news_tracker(items)
        return {"news": items}

    def _get_positions(args: dict) -> dict:
        return {"equity": broker.equity, "positions": broker.positions()}

    def _place_paper_order(args: dict) -> dict:
        if order_preparer:
            args = order_preparer(args)
        ok, reason = guardrail_check(args)
        if not ok:
            return {"executed": False, "rejected_reason": reason}
        try:
            trade = broker.place_order(
                symbol=args["symbol"], side=args["side"],
                notional_usd=float(args["notional_usd"]), rationale=args.get("rationale", ""),
                stop_pct=args.get("stop_pct"), target_pct=args.get("target_pct"),
            )
        except Exception as exc:
            return {"executed": False, "rejected_reason": f"execution failed: {exc}"}
        return {"executed": True, "trade_id": trade.id, "entry_price": trade.entry_price,
                "stop_loss": trade.stop_loss, "take_profit": trade.take_profit}

    def _close_paper_position(args: dict) -> dict:
        symbol = args.get("symbol")
        if symbol not in broker.open_trades:
            return {"executed": False, "rejected_reason": f"no open position on '{symbol}'"}
        try:
            trade = broker.close_position(symbol, reason=f"llm: {args.get('rationale', '')}")
        except Exception as exc:
            return {"executed": False, "rejected_reason": f"close failed: {exc}"}
        if trade is None:
            return {"executed": False, "rejected_reason": f"no open position on '{symbol}'"}
        return {"executed": True, "action": "close", "exit_price": trade.exit_price,
                "pnl": trade.pnl, "pnl_pct": trade.pnl_pct}

    def _hold(args: dict) -> dict:
        return {"executed": True, "action": "hold", "rationale": args.get("rationale", "")}

    return {
        "get_market_context": _get_market_context,
        "get_news_digest": _get_news_digest,
        "get_positions": _get_positions,
        "place_paper_order": _place_paper_order,
        "close_paper_position": _close_paper_position,
        "hold": _hold,
    }
