"""
Shadow ledger: every candidate is followed as if it had been taken — with the
same fill costs, stop and target as a real paper trade — whatever the judge
decided. That is what makes the judge measurable: compare the outcome of
approved vs vetoed candidates on the same footing.

Live, the book is updated with price snapshots (fills at the observed price,
like the real exit monitor). In replay it gets bars (fills at the stop/target
level, stop first if both were touched — the conservative convention used by
backend/strategy/intraday_setups.simulate_setups).
"""
from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from typing import Dict, List, Optional

from backend.agent.paper_execution import fill_entry_price, fill_exit_price
from backend.hunters.base import Candidate

logger = logging.getLogger(__name__)

TAKEN = ("approved", "reduced")
JUDGED = ("approved", "reduced", "vetoed")


class ShadowBook:
    def __init__(self, repo=None, max_hold_sec: int = 24 * 3600):
        self.repo = repo
        self.max_hold_ms = max_hold_sec * 1000
        self.open_rows: Dict[str, dict] = {}
        self._lock = threading.RLock()

    def _persist(self, row: dict):
        if self.repo:
            self.repo.save(row)

    def restore(self, rows: List[dict]):
        with self._lock:
            self.open_rows = {r["id"]: r for r in rows}

    def has(self, cand_id: str) -> bool:
        return cand_id in self.open_rows

    def open(self, cand: Candidate, decision: str, reason: str, raw_price: float,
             now_ms: int, live_trade_id: Optional[str] = None) -> dict:
        d = cand.direction
        entry = fill_entry_price(raw_price, d)
        row = {
            "id": cand.id, "hunter": cand.hunter, "version": cand.version, "symbol": cand.symbol,
            "direction": d, "created_at": now_ms, "entry_price": entry,
            "stop_price": entry * (1 - d * cand.stop_pct),
            "target_price": entry * (1 + d * cand.target_pct),
            "stop_pct": cand.stop_pct, "decision": decision, "decision_reason": (reason or "")[:1000],
            "context": json.dumps(cand.context), "status": "open", "live_trade_id": live_trade_id,
        }
        with self._lock:
            self.open_rows[cand.id] = row
        self._persist(row)
        return row

    def _close(self, row: dict, raw_exit: float, reason: str, now_ms: int) -> dict:
        d = row["direction"]
        exit_price = fill_exit_price(raw_exit, d)
        pnl_pct = (exit_price - row["entry_price"]) / row["entry_price"] * d
        row.update(status="closed", exit_time=now_ms, exit_price=exit_price, exit_reason=reason,
                   r_multiple=pnl_pct / row["stop_pct"])
        self._persist(row)
        return row

    def update(self, symbol: str, now_ms: int, price: Optional[float] = None,
               high: Optional[float] = None, low: Optional[float] = None,
               close: Optional[float] = None) -> List[dict]:
        """Price snapshot (`price`) or a bar (`high`/`low`/`close`)."""
        tick = price is not None
        if tick:
            high = low = close = price
        closed = []
        with self._lock:
            for cid, row in list(self.open_rows.items()):
                if row["symbol"] != symbol:
                    continue
                d, stop, target = row["direction"], row["stop_price"], row["target_price"]
                hit_stop = low <= stop if d == 1 else high >= stop
                hit_target = high >= target if d == 1 else low <= target
                if hit_stop:
                    closed.append(self._close(row, price if tick else stop, "stop", now_ms))
                elif hit_target:
                    closed.append(self._close(row, price if tick else target, "target", now_ms))
                elif now_ms - row["created_at"] >= self.max_hold_ms:
                    closed.append(self._close(row, close, "time", now_ms))
                else:
                    continue
                del self.open_rows[cid]
        return closed


def _summary(rs: List[float]) -> dict:
    n = len(rs)
    return {"n": n,
            "mean_r": round(sum(rs) / n, 4) if n else None,
            "win_rate": round(sum(1 for r in rs if r > 0) / n, 4) if n else None,
            "total_r": round(sum(rs), 4)}


def shadow_stats(closed: List[dict]) -> dict:
    """Per-hunter outcome by decision, plus the judge's value-add:
    mean R of what it let through minus mean R of everything it judged.
    Positive = the LLM filter is helping; needs a decent n before it means
    anything."""
    by_hunter: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in closed:
        r = row.get("r_multiple")
        if r is None:
            continue
        by_hunter[row["hunter"]]["all"].append(r)
        by_hunter[row["hunter"]][row["decision"].split(":")[0]].append(r)

    hunters = {h: {k: _summary(v) for k, v in groups.items()} for h, groups in by_hunter.items()}
    judged = [r["r_multiple"] for r in closed if r.get("r_multiple") is not None
              and r["decision"] in JUDGED]
    taken = [r["r_multiple"] for r in closed if r.get("r_multiple") is not None
             and r["decision"] in TAKEN]
    vetoed = [r["r_multiple"] for r in closed if r.get("r_multiple") is not None
              and r["decision"] == "vetoed"]
    judge = {
        "judged": _summary(judged), "taken": _summary(taken), "vetoed": _summary(vetoed),
        "value_add_r": (round(_summary(taken)["mean_r"] - _summary(judged)["mean_r"], 4)
                        if taken and judged else None),
        "veto_precision": (round(sum(1 for r in vetoed if r <= 0) / len(vetoed), 4)
                           if vetoed else None),
    }
    return {"hunters": hunters, "judge": judge}
