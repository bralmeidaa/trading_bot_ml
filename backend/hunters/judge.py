"""
LLM judge for hunter candidates. The rules decide WHAT to trade and where
the stop/target go; the LLM only approves, halves (reduce) or vetoes, using
what the rules can't see: news, the wider market, the book's exposure.

One LLM call per batch of candidates. The verdicts must come back through the
`submit_verdicts` tool; anything missing or malformed is "failed" (fail
closed: never executed). A failure is kept apart from a real veto so it
doesn't pollute the measurement of the judge's vetoes.
"""
from __future__ import annotations

import json
import logging
from typing import Callable, Dict, List, Tuple

from backend.agent.tools import TOOL_SCHEMAS
from backend.hunters.base import Candidate

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are the risk judge for a set of mechanical crypto \
trading strategies (paper trading). Each candidate below comes from a fixed \
rule with its own stop-loss and take-profit already set — you cannot change \
them. For every candidate decide:
- approve: take it at normal size;
- reduce: take it at half size;
- veto: skip it.

Use what the rule can't see: fresh news (get_news_digest; items with \
cycles_since_first_seen > 0 are already known), the wider market move \
(get_market_context) and the book's open positions (get_positions — \
avoid stacking same-direction bets on coins that move together). Approve \
when nothing you find argues against the trade; veto or reduce only for a \
concrete reason and name it. Your verdicts are scored afterwards against \
what each candidate actually did, including the vetoed ones.

Finish by calling submit_verdicts once with one verdict per candidate id."""

VERDICT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_verdicts",
        "description": "Submit the final verdict for every candidate (call once).",
        "parameters": {
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "decision": {"type": "string", "enum": ["approve", "reduce", "veto"]},
                            "rationale": {"type": "string"},
                        },
                        "required": ["candidate_id", "decision", "rationale"],
                    },
                },
            },
            "required": ["verdicts"],
        },
    },
}

_READ_TOOLS = ("get_market_context", "get_news_digest", "get_positions")
_DECISION = {"approve": "approved", "reduce": "reduced", "veto": "vetoed"}


def system_prompt(lessons: List[str] = None) -> str:
    """JUDGE_PROMPT plus the student's active lessons — each one already
    verified on >= 30 closed shadow trades (backend/student)."""
    if not lessons:
        return JUDGE_PROMPT
    return (JUDGE_PROMPT + "\n\nLessons measured on this system's own past candidates "
            "(use them as evidence, not as rules):\n" + "\n".join(f"- {l}" for l in lessons))


def judge_tools() -> List[dict]:
    return [t for t in TOOL_SCHEMAS if t["function"]["name"] in _READ_TOOLS] + [VERDICT_SCHEMA]


def describe(c: Candidate) -> dict:
    return {"candidate_id": c.id, "strategy": c.hunter, "symbol": c.symbol,
            "side": "long" if c.direction == 1 else "short", "price": c.ref_price,
            "stop_pct": round(c.stop_pct, 4), "target_pct": round(c.target_pct, 4),
            "why_the_rule_fired": c.rationale}


def run_judge(llm_client, candidates: List[Candidate],
              read_handlers: Dict[str, Callable[[dict], dict]],
              max_tool_turns: int = 6,
              lessons: List[str] = None) -> Tuple[Dict[str, Tuple[str, str]], object]:
    """Returns ({candidate_id: (decision, rationale)}, ChatResult). Every
    candidate gets a verdict; failures become ('failed', reason)."""
    submitted: Dict[str, Tuple[str, str]] = {}
    ids = {c.id for c in candidates}

    def _submit(args: dict) -> dict:
        for v in args.get("verdicts") or []:
            cid, dec = v.get("candidate_id"), _DECISION.get(v.get("decision"))
            if cid in ids and dec:
                submitted[cid] = (dec, str(v.get("rationale", ""))[:1000])
        return {"accepted": len(submitted), "expected": len(ids)}

    dispatch = {name: read_handlers[name] for name in _READ_TOOLS if name in read_handlers}
    dispatch["submit_verdicts"] = _submit
    messages = [
        {"role": "system", "content": system_prompt(lessons)},
        {"role": "user", "content": "Candidates:\n" + json.dumps([describe(c) for c in candidates], indent=1)},
    ]
    result = llm_client.chat_with_tools(messages, judge_tools(), dispatch, max_tool_turns=max_tool_turns)
    reason = f"judge error: {result.error}" if result.error else "no verdict returned by judge"
    verdicts = {cid: submitted.get(cid, ("failed", reason)) for cid in ids}
    return verdicts, result
