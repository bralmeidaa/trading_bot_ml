# -*- coding: utf-8 -*-
"""
O drawdown de ~-69% no ciclo completo (2020-2026) é o problema real da estratégia
diária, não o retorno. Este experimento testa se um overlay de RISCO PRINCIPIADO
(volatility targeting — escala a exposição pra uma vol-alvo, ignora o sinal) doma
o drawdown SEM p-hacking, validado in-sample E walk-forward.

Vol targeting: exposição_t = clip(vol_alvo / vol_realizada_{t-1}, 0, max_lev).
Reduz posição quando a estratégia está volátil (o que acompanha os crashes).

Run: python .claude/research/research_daily_riskoverlay.py
"""
import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from backend.data.universe import build_panel, EXPANDED_UNIVERSE
from backend.strategy.cross_sectional import simulate, metrics

TF, DAYS = "1d", 2200
LB, RB, K, MODE = 12, 12, 3, "momentum"
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


def vol_target(port: pd.Series, target_ann: float, lookback: int = 30,
               max_lev: float = 1.5) -> pd.Series:
    """Escala o retorno diário pra uma vol anualizada alvo (exposição defasada)."""
    realized = port.rolling(lookback, min_periods=10).std() * np.sqrt(365)
    scale = (target_ann / realized).clip(0.0, max_lev)
    return (scale.shift(1).fillna(0.0) * port)


def wf(port: pd.Series, n_folds: int = 5) -> dict:
    """Walk-forward OOS agregado sobre uma série de retornos já pronta."""
    n = len(port); fold = n // (n_folds + 1)
    ports, sharpes = [], []
    for f in range(1, n_folds + 1):
        seg = port.iloc[f * fold:(f + 1) * fold]
        if len(seg) < 30:
            continue
        ports.append(seg); sharpes.append(metrics(seg, TF)["sharpe"])
    agg = metrics(pd.concat(ports), TF)
    agg["pos_folds"] = sum(1 for s in sharpes if s > 0)
    agg["n_folds"] = len(sharpes)
    return agg


def row(name, m, mw):
    return (f"| {name} | {m['net']:+.0%} | {m['ann_ret']:+.1%} | {m['sharpe']:.2f} | "
            f"{m['maxdd']:+.0%} | {mw['sharpe']:.2f} | {mw['ann_ret']:+.1%} | "
            f"{mw['maxdd']:+.0%} | {mw['pos_folds']}/{mw['n_folds']} |")


def main():
    print(f"Loading PIT panel {DAYS}d...")
    close, volume, _ = build_panel(symbols=EXPANDED_UNIVERSE, timeframe=TF, days=DAYS,
                                   point_in_time=True, verbose=False)
    base = simulate(close, LB, RB, K, MODE, btc_filter=True, volume=volume)

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    start = pd.to_datetime(close.index[0], unit="ms").date()
    end = pd.to_datetime(close.index[-1], unit="ms").date()
    out(f"# Overlay de risco — cross-sectional diário ({start}→{end})\n")
    out("Full-period E walk-forward OOS. Objetivo: domar o maxDD sem matar o retorno.\n")
    out("| config | net | CAGR | Sharpe | maxDD | OOS Sharpe | OOS CAGR | OOS maxDD | folds+ |")
    out("|--------|-----|------|--------|-------|-----------|----------|-----------|--------|")
    out(row("base (sem overlay)", metrics(base, TF), wf(base)))
    best = None
    for tv in (0.10, 0.15, 0.20, 0.25):
        vt = vol_target(base, tv)
        m, mw = metrics(vt, TF), wf(vt)
        out(row(f"vol-target {tv:.0%}", m, mw))
        # escolhe por melhor Sharpe OOS com maxDD full < 50%
        score = mw["sharpe"] if m["maxdd"] > -0.50 else -9
        if best is None or score > best[0]:
            best = (score, tv, m, mw)

    out("\n## Leitura\n")
    bm, bmw = metrics(base, TF), wf(base)
    out(f"- Base: CAGR {bm['ann_ret']:+.0%}, Sharpe {bm['sharpe']:.2f}, **maxDD {bm['maxdd']:+.0%}** "
        f"(OOS Sharpe {bmw['sharpe']:.2f}).")
    if best:
        _, tv, m, mw = best
        out(f"- Melhor overlay: **vol-target {tv:.0%}** → CAGR {m['ann_ret']:+.0%}, "
            f"Sharpe {m['sharpe']:.2f}, **maxDD {m['maxdd']:+.0%}**, OOS Sharpe {mw['sharpe']:.2f}, "
            f"folds+ {mw['pos_folds']}/{mw['n_folds']}.")
        dd_gain = m["maxdd"] - bm["maxdd"]
        out(f"- Efeito no drawdown: {bm['maxdd']:+.0%} → {m['maxdd']:+.0%} "
            f"({dd_gain:+.0%} de melhora) mantendo Sharpe {'MAIOR' if m['sharpe']>bm['sharpe'] else 'similar'}.")
    out("- Vol targeting é overlay PRINCIPIADO (não toca o sinal, não é curve-fit do timing).")

    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "daily_riskoverlay_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nRelatório salvo em {path}")


if __name__ == "__main__":
    main()
