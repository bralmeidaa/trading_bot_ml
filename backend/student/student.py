"""
The LLM student: studies the system's own results and proposes changes that
only go live if the data backs them.

Weekly (full) cycle:
  1. replay the seeded hunters on the TRAINING segment (older history);
  2. build a bucketed dossier: training replay + live shadow ledger;
  3. one LLM call → up to 5 hypotheses (variant specs) and lessons;
  4. every hypothesis is replayed on the TEST segment (the most recent
     `test_days`, never shown in the dossier) next to its parent. It passes
     only with n ≥ 30, mean R > 0, beating the parent by a margin, positive
     in all-but-one fold and P(edge > 0) high enough. Passed → status
     "shadow": it joins the tournament with ZERO capital; the allocator gives
     it capital only after 30 forward judge-approved shadow trades say so;
  5. lessons are checked against the shadow ledger (n ≥ 30, posterior
     confidence ≥ 0.8); confirmed ones go into the judge's prompt (max 10).
Daily (light) cycle: retire variants that lose forward, re-check lessons,
refresh the tournament allocation. Every promotion, retirement and new
lesson is emailed.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from backend.allocator.thompson import p_positive
from backend.hunters.base import Hunter
from backend.hunters.shadow import TAKEN
from backend.hunters.strategies import BASE_HUNTERS
from backend.replay.report import max_drawdown_r, strategy_report
from backend.student.dossier import build_dossier, shadow_as_trades
from backend.student.variants import (NOT_REPLAYABLE, build_variant, passes, spec_hash,
                                      validate_filters, validate_spec)

logger = logging.getLogger(__name__)

DAY_MS = 86_400_000

STUDENT_PROMPT = """You are the research student of a crypto paper-trading \
system. Mechanical strategies ("hunters") propose trades with their own \
stop/target; an LLM judge approves or vetoes them. You get bucketed \
statistics of their results (R = profit in units of the risk taken, net of \
costs) and propose improvements.

You may propose up to 5 HYPOTHESES, each a variant of one hunter:
- change parameters, only those listed for that hunter and within min/max;
- and/or add up to 3 filters using ONLY the listed features and ops.
Every hypothesis is then tested on recent data you have NOT seen, against \
its parent. It must reach >= 30 trades there, so do not filter so narrowly \
that almost nothing is left; a filter that keeps a bucket with n < 30 in \
the training stats will almost surely fail. Prefer few, well-argued \
hypotheses grounded in a bucket that is clearly worse or better than the \
rest. Don't repeat an existing variant.

You may also propose up to 5 LESSONS for the judge: a pattern in the LIVE \
shadow ledger, stated as a short sentence plus a machine-checkable spec \
(hunter or null for all, filters, claim "loses" or "wins"). A lesson is \
only kept if the ledger confirms it with >= 30 cases.

If nothing in the data justifies a change, propose nothing. Finish by \
calling submit_proposals exactly once."""

_FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "feature": {"type": "string"},
        "op": {"type": "string", "enum": ["<", ">", "between", "in"]},
        "value": {},
    },
    "required": ["feature", "op", "value"],
}

PROPOSALS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_proposals",
        "description": "Submit hypotheses (variants) and lessons. Call once.",
        "parameters": {
            "type": "object",
            "properties": {
                "hypotheses": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "parent": {"type": "string"},
                        "params": {"type": "object"},
                        "filters": {"type": "array", "items": _FILTER_SCHEMA},
                        "rationale": {"type": "string"},
                    },
                    "required": ["parent", "rationale"]}},
                "lessons": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "hunter": {"type": ["string", "null"]},
                        "filters": {"type": "array", "items": _FILTER_SCHEMA},
                        "claim": {"type": "string", "enum": ["loses", "wins"]},
                    },
                    "required": ["text", "filters", "claim"]}},
            },
            "required": ["hypotheses", "lessons"],
        },
    },
}


@dataclass
class StudentConfig:
    replay_days: int = 120
    test_days: int = 40
    max_hypotheses: int = 5
    max_active_variants: int = 6
    min_trades: int = 30
    min_improvement_r: float = 0.05
    min_p_positive: float = 0.75
    retire_p_below: float = 0.3
    retire_dd_r: float = -10.0
    max_lessons: int = 10
    lesson_min_n: int = 30
    lesson_min_confidence: float = 0.8
    max_tool_turns: int = 3


def evaluate(variant_trades: List[dict], parent_trades: List[dict], days: float,
             cfg: StudentConfig) -> tuple:
    """(passed, reason, evidence) for a variant vs its parent on the same data."""
    v, p = strategy_report(variant_trades, days), strategy_report(parent_trades, days)
    evidence = {"variant": v, "parent": p}
    if v["n"] < cfg.min_trades:
        return False, f"only {v['n']} trades out of sample (< {cfg.min_trades})", evidence
    if v["mean_r"] <= 0:
        return False, f"mean {v['mean_r']:+.3f}R out of sample is not positive", evidence
    bar = (p["mean_r"] if p["mean_r"] is not None else 0.0) + cfg.min_improvement_r
    if v["mean_r"] < bar:
        return False, (f"mean {v['mean_r']:+.3f}R does not beat the parent "
                       f"({p['mean_r']}R) by {cfg.min_improvement_r}R"), evidence
    fc = v["folds"]
    if not (fc["n_folds"] >= 3 and fc["pos_folds"] >= fc["n_folds"] - 1):
        return False, f"inconsistent across folds {fc['per_fold']}", evidence
    if v["p_positive"] < cfg.min_p_positive:
        return False, f"P(edge>0)={v['p_positive']} < {cfg.min_p_positive}", evidence
    return True, (f"out of sample: n={v['n']}, {v['mean_r']:+.3f}R/trade vs parent "
                  f"{p['mean_r']}R, folds {fc['per_fold']}, P(edge>0)={v['p_positive']}"), evidence


def verify_lesson(spec: dict, shadow_trades: List[dict]) -> dict:
    rs = [t["r"] for t in shadow_trades
          if (spec.get("hunter") in (None, t["hunter"]))
          and passes(spec.get("filters") or [], t["context"], t["direction"])]
    n = len(rs)
    pp = p_positive(rs)
    conf = pp if spec.get("claim") == "wins" else 1 - pp
    return {"n": n, "mean_r": round(float(np.mean(rs)), 4) if n else 0.0,
            "confidence": round(conf, 4)}


def lesson_text(text: str, check: dict) -> str:
    return f"{text.strip()} [n={check['n']}, {check['mean_r']:+.2f}R/trade]"


class Student:
    def __init__(self, llm_client, shadow_repo, variant_repo, lesson_repo, engine=None,
                 notifier: Optional[Callable[[str, str], None]] = None,
                 replay_fn: Optional[Callable[[List[Hunter], int, int], List[dict]]] = None,
                 cfg: Optional[StudentConfig] = None, now_fn: Callable[[], float] = time.time):
        self.llm_client = llm_client
        self.shadow_repo = shadow_repo
        self.variant_repo = variant_repo
        self.lesson_repo = lesson_repo
        self.engine = engine
        self.notifier = notifier
        self.replay_fn = replay_fn or self._default_replay
        self.cfg = cfg or StudentConfig()
        self.now_fn = now_fn
        self.last_daily: Optional[dict] = None
        self.last_weekly: Optional[dict] = None

    @staticmethod
    def _default_replay(hunters, start_ms, end_ms):
        from backend.replay.run import run_replay
        return run_replay(hunters, start_ms, end_ms)["trades"]

    def _notify(self, subject: str, body: str):
        logger.info(f"student: {subject}\n{body}")
        if self.notifier:
            try:
                self.notifier(subject, body)
            except Exception as exc:
                logger.warning(f"student notice failed: {exc}")

    # ── engine wiring ────────────────────────────────────────────────────
    def active_variants(self) -> List[Hunter]:
        out = []
        for row in self.variant_repo.list(status="shadow"):
            try:
                out.append(build_variant(row))
            except Exception as exc:
                logger.warning(f"student: cannot build variant {row.get('id')}: {exc}")
        return out

    def active_lessons(self) -> List[str]:
        return [l["text"] for l in self.lesson_repo.list(status="active")]

    def apply_to_engine(self):
        if self.engine is not None:
            self.engine.set_variants(self.active_variants())
            self.engine.lessons = self.active_lessons()

    # ── daily ────────────────────────────────────────────────────────────
    def daily(self) -> dict:
        shadow = shadow_as_trades(self.shadow_repo.get_closed())
        retired, lessons_retired = [], []
        for row in self.variant_repo.list(status="shadow"):
            rs = [t["r"] for t in shadow if t["hunter"] == row["id"] and t["decision"] in TAKEN]
            pp, dd = p_positive(rs), max_drawdown_r(rs)
            reason = None
            if len(rs) >= self.cfg.min_trades and pp < self.cfg.retire_p_below:
                reason = f"forward shadow n={len(rs)}, P(edge>0)={pp:.2f} < {self.cfg.retire_p_below}"
            elif dd <= self.cfg.retire_dd_r:
                reason = f"forward shadow drawdown {dd:.1f}R <= {self.cfg.retire_dd_r}R"
            if reason:
                row.update(status="retired", status_note=reason)
                self.variant_repo.save(row)
                retired.append((row["id"], reason))
        for les in self.lesson_repo.list(status="active"):
            chk = verify_lesson(les["spec"], shadow)
            if chk["n"] < self.cfg.lesson_min_n or chk["confidence"] < self.cfg.lesson_min_confidence:
                les.update(status="retired", n=chk["n"], mean_r=chk["mean_r"],
                           confidence=chk["confidence"])
                self.lesson_repo.save(les)
                lessons_retired.append(les["text"])
            else:
                base = les["text"].split(" [n=")[0]
                les.update(text=lesson_text(base, chk), n=chk["n"], mean_r=chk["mean_r"],
                           confidence=chk["confidence"])
                self.lesson_repo.save(les)
        if retired or lessons_retired:
            self._notify(f"[student] retired {len(retired)} variant(s), {len(lessons_retired)} lesson(s)",
                         "\n".join([f"variant {vid}: {why}" for vid, why in retired]
                                   + [f"lesson no longer supported: {t}" for t in lessons_retired]))
        self.apply_to_engine()
        if self.engine is not None:
            self.engine.refresh_allocation(notify=True)
        self.last_daily = {"at": self.now_fn(), "retired_variants": retired,
                           "retired_lessons": lessons_retired}
        return self.last_daily

    # ── weekly ───────────────────────────────────────────────────────────
    def _ask(self, dossier: dict) -> tuple:
        got: Dict[str, list] = {}

        def _submit(args: dict) -> dict:
            got["hypotheses"] = list(args.get("hypotheses") or [])[: self.cfg.max_hypotheses]
            got["lessons"] = list(args.get("lessons") or [])[:5]
            return {"received": {k: len(v) for k, v in got.items()}}

        messages = [{"role": "system", "content": STUDENT_PROMPT},
                    {"role": "user", "content": json.dumps(dossier, default=str)}]
        result = self.llm_client.chat_with_tools(messages, [PROPOSALS_SCHEMA],
                                                 {"submit_proposals": _submit},
                                                 max_tool_turns=self.cfg.max_tool_turns)
        return got.get("hypotheses", []), got.get("lessons", []), result

    def weekly(self) -> dict:
        cfg = self.cfg
        end = int(self.now_fn() * 1000) // DAY_MS * DAY_MS
        split, start = end - cfg.test_days * DAY_MS, end - cfg.replay_days * DAY_MS
        parents = [f() for n, f in BASE_HUNTERS.items() if n not in NOT_REPLAYABLE]
        train = self.replay_fn(parents, start, split)
        existing = self.variant_repo.list()
        lessons = self.lesson_repo.list()
        shadow_rows = self.shadow_repo.get_closed()
        dossier = build_dossier(train, (split - start) / DAY_MS, shadow_rows, existing, lessons)
        hyps, lesson_props, llm = self._ask(dossier)
        out = {"at": self.now_fn(), "window": {"train": [start, split], "test": [split, end]},
               "llm_error": getattr(llm, "error", None),
               "tokens": (getattr(llm, "prompt_tokens", 0) or 0) + (getattr(llm, "completion_tokens", 0) or 0),
               "hypotheses": [], "lessons": []}

        # ── hypotheses → replay on the unseen test segment ──
        seen = {v["spec_hash"] for v in existing}
        counts: Dict[str, int] = {}
        for v in existing:
            counts[v["parent"]] = counts.get(v["parent"], 0) + 1
        candidates = []
        for raw in hyps:
            spec, err = validate_spec(raw)
            if err:
                out["hypotheses"].append({"spec": raw, "status": "invalid", "reason": err})
                continue
            h = spec_hash(spec)
            if h in seen:
                out["hypotheses"].append({"spec": spec, "status": "duplicate"})
                continue
            seen.add(h)
            counts[spec["parent"]] = counts.get(spec["parent"], 0) + 1
            vid = f"{spec['parent']}~v{counts[spec['parent']]}"
            candidates.append((vid, h, spec))
        if candidates:
            variants = [build_variant({"id": vid, **spec}) for vid, _, spec in candidates]
            involved = sorted({spec["parent"] for _, _, spec in candidates})
            test = self.replay_fn([BASE_HUNTERS[p]() for p in involved] + variants, split, end)
            by: Dict[str, List[dict]] = {}
            for t in test:
                by.setdefault(t["hunter"], []).append(t)
            days = (end - split) / DAY_MS
            results = []
            for vid, h, spec in candidates:
                ok, reason, ev = evaluate(by.get(vid, []), by.get(spec["parent"], []), days, cfg)
                results.append([vid, h, spec, ok, reason, ev])
            slots = cfg.max_active_variants - len(self.variant_repo.list(status="shadow"))
            passed = sorted([r for r in results if r[3]], key=lambda r: -r[5]["variant"]["mean_r"])
            for r in passed[max(0, slots):]:
                r[3], r[4] = False, f"passed ({r[4]}) but no free variant slot"
            promoted = []
            for vid, h, spec, ok, reason, ev in results:
                row = {"id": vid, "parent": spec["parent"], "spec_hash": h, "params": spec["params"],
                       "filters": spec["filters"], "rationale": spec["rationale"],
                       "status": "shadow" if ok else "rejected", "evidence": ev, "status_note": reason}
                self.variant_repo.save(row)
                out["hypotheses"].append({"id": vid, "spec": spec, "status": row["status"], "reason": reason})
                if ok:
                    promoted.append(f"{vid} ({spec['rationale'][:200]})\n  params {spec['params']} "
                                    f"filters {spec['filters']}\n  {reason}")
            if promoted:
                self._notify(f"[student] {len(promoted)} new variant(s) in shadow",
                             "Entered the tournament with zero capital; the allocator "
                             "funds them only after 30 forward shadow trades.\n\n" + "\n\n".join(promoted))

        # ── lessons → verified on the shadow ledger ──
        shadow = shadow_as_trades(shadow_rows)
        active = self.lesson_repo.list(status="active")
        new_lessons = []
        for raw in lesson_props:
            filters, err = validate_filters(raw.get("filters") or [])
            spec = {"hunter": raw.get("hunter"), "filters": filters or [], "claim": raw.get("claim")}
            if not err and spec["claim"] not in ("loses", "wins"):
                err = "claim must be loses or wins"
            if not err and not str(raw.get("text", "")).strip():
                err = "empty text"
            if err:
                out["lessons"].append({"text": raw.get("text"), "status": "invalid", "reason": err})
                continue
            chk = verify_lesson(spec, shadow)
            ok = chk["n"] >= cfg.lesson_min_n and chk["confidence"] >= cfg.lesson_min_confidence
            status = "active" if ok else "unsupported"
            out["lessons"].append({"text": raw.get("text"), "status": status, **chk})
            if ok:
                new_lessons.append({"id": spec_hash({"parent": spec["hunter"] or "*", "params": {},
                                                      "filters": spec["filters"]}) + spec["claim"][0],
                                    "text": lesson_text(str(raw.get("text", "")), chk), "spec": spec,
                                    "status": "active", **chk})
        if new_lessons:
            pool = {l["id"]: l for l in active}
            pool.update({l["id"]: l for l in new_lessons})
            ranked = sorted(pool.values(), key=lambda l: -l["confidence"])
            for i, les in enumerate(ranked):
                les["status"] = "active" if i < cfg.max_lessons else "retired"
                self.lesson_repo.save(les)
            kept = [l for l in new_lessons if l["status"] == "active"]
            if kept:
                self._notify(f"[student] {len(kept)} new lesson(s) for the judge",
                             "\n".join(l["text"] for l in kept))
        self.last_weekly = out
        self.daily()
        return out
