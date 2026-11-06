# -*- coding: utf-8 -*-
"""
Pesquisa: variante de regime que opere MAIS (menos flat) sem piorar o risco OOS.

O filtro atual (BTC EMA12 > EMA60) deixa a carteira flat ~45% do tempo. Removê-lo
estoura o OOS maxDD (-43% -> -88%). Aqui testamos variantes PRINCIPIADAS (spans de
EMA, histerese, slope) a exposure 0.35, medindo %do tempo investido vs risco OOS.

Gate honesto: só vale trocar se DOMINAR o atual — mais %investido E OOS maxDD nao
pior (<= ~-45%) E OOS Sharpe >= ~0.9 E >=4/5 folds positivos. Senao, o atual fica.

Run: python .claude/research/research_regime.py
"""
import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from backend.data.universe import build_panel, EXPANDED_UNIVERSE
from backend.strategy.cross_sectional import simulate, metrics

TF, DAYS = "1d", 2200
LB, RB, K = 12, 12, 3
EXPO = 0.35
FOLDS = 5
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


# ── regime variants (each returns a bool series over close.index) ──
def ema_cross(close, fast, slow, band=0.0):
    """BTC EMA(fast) vs EMA(slow). band>0 = hysteresis (ON >+band, OFF <-band)."""
    btc = close["BTC/USDT"]
    ef, es = btc.ewm(span=fast).mean(), btc.ewm(span=slow).mean()
    if band <= 0:
        return ef > es
    gap = (ef / es - 1.0).values
    on, cur = [], False
    for g in gap:
        if not cur and g > band:
            cur = True
        elif cur and g < -band:
            cur = False
        on.append(cur)
    return pd.Series(on, index=close.index)


def ema_slope(close, span):
    """BTC EMA(span) subindo (derivada > 0)."""
    return close["BTC/USDT"].ewm(span=span).mean().diff() > 0


VARIANTS = [
    ("atual EMA12/60",      lambda c: ema_cross(c, 12, 60, 0.0)),
    ("EMA8/40",             lambda c: ema_cross(c, 8, 40, 0.0)),
    ("EMA10/30",            lambda c: ema_cross(c, 10, 30, 0.0)),
    ("EMA10/40",            lambda c: ema_cross(c, 10, 40, 0.0)),
    ("EMA15/45",            lambda c: ema_cross(c, 15, 45, 0.0)),
    ("EMA20/50",            lambda c: ema_cross(c, 20, 50, 0.0)),
    ("EMA20/100",           lambda c: ema_cross(c, 20, 100, 0.0)),
    ("EMA12/60 band1%",     lambda c: ema_cross(c, 12, 60, 0.01)),
    ("EMA12/60 band2%",     lambda c: ema_cross(c, 12, 60, 0.02)),
    ("slope EMA30",         lambda c: ema_slope(c, 30)),
    ("slope EMA50",         lambda c: ema_slope(c, 50)),
]


def pct_on(close, regime_fn):
    return float(regime_fn(close).iloc[LB:].mean())


def full(close, vol, regime_fn):
    p = EXPO * simulate(close, LB, RB, K, volume=vol, regime_override=regime_fn(close))
    return metrics(p, TF)


def wf(close, vol, regime_fn):
    """Walk-forward: regime recomputado em cada fold OOS (sem leakage)."""
    n = len(close); f = n // (FOLDS + 1)
    ports, sharpes = [], []
    for i in range(1, FOLDS + 1):
        seg, vseg = close.iloc[i * f:(i + 1) * f], vol.iloc[i * f:(i + 1) * f]
        if len(seg) < LB + RB + 5:
            continue
        p = EXPO * simulate(seg, LB, RB, K, volume=vseg, regime_override=regime_fn(seg))
        if len(p):
            ports.append(p); sharpes.append(metrics(p, TF)["sharpe"])
    agg = metrics(pd.concat(ports), TF)
    agg["pos"] = sum(1 for s in sharpes if s > 0); agg["nf"] = len(sharpes)
    return agg


def main():
    print(f"Loading PIT panel {DAYS}d...")
    close, vol, _ = build_panel(symbols=EXPANDED_UNIVERSE, timeframe=TF, days=DAYS,
                                point_in_time=True, verbose=False)
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    start = pd.to_datetime(close.index[0], unit="ms").date()
    end = pd.to_datetime(close.index[-1], unit="ms").date()
    out(f"# Regime — variantes menos-flat (exposure {EXPO:.0%}, {start}→{end})\n")
    out("Gate: DOMINAR o atual — mais %investido E OOS maxDD ≤ ~-45% E OOS Sharpe ≥ ~0.9 E ≥4/5 folds.\n")
    out("| variante | %investido | CAGR | Sharpe | maxDD | OOS Sharpe | OOS maxDD | folds+ |")
    out("|---|---|---|---|---|---|---|---|")
    rows = {}
    for name, fn in VARIANTS:
        po = pct_on(close, fn)
        m, w = full(close, vol, fn), wf(close, vol, fn)
        rows[name] = (po, m, w)
        out(f"| {name} | {po:.0%} | {m['ann_ret']:+.1%} | {m['sharpe']:.2f} | {m['maxdd']:+.0%} "
            f"| {w['sharpe']:.2f} | {w['maxdd']:+.0%} | {w['pos']}/{w['nf']} |")

    base_on, base_m, base_w = rows["atual EMA12/60"]
    out("\n## Veredito\n")

    # (1) "menos flat" real: >=3pp mais investido sem piorar risco OOS
    less_flat = [(n, po, m, w) for n, (po, m, w) in rows.items() if n != "atual EMA12/60"
                 and po > base_on + 0.03 and w["maxdd"] >= -0.45 and w["sharpe"] >= 0.90 and w["pos"] >= 4]
    # (2) refino de QUALIDADE: %investido semelhante, mas Pareto melhor (full Sharpe
    #     maior, folds >= base, OOS Sharpe e OOS maxDD nao piores)
    quality = [(n, po, m, w) for n, (po, m, w) in rows.items() if n != "atual EMA12/60"
               and m["sharpe"] > base_m["sharpe"] + 0.03 and w["pos"] >= base_w["pos"]
               and w["sharpe"] >= base_w["sharpe"] - 0.02 and w["maxdd"] >= base_w["maxdd"] - 0.01]

    out(f"Atual (EMA12/60): {base_on:.0%} investido, full Sharpe {base_m['sharpe']:.2f}, "
        f"OOS Sharpe {base_w['sharpe']:.2f}, OOS maxDD {base_w['maxdd']:+.0%}, folds {base_w['pos']}/{base_w['nf']}.\n")

    if less_flat:
        out("**MENOS FLAT** (>=3pp mais investido sem piorar risco OOS):")
        for n, po, m, w in sorted(less_flat, key=lambda x: -x[1]):
            out(f"- {n}: {po:.0%} investido, OOS Sharpe {w['sharpe']:.2f}, OOS maxDD {w['maxdd']:+.0%}.")
    else:
        out("**MENOS FLAT: nenhuma.** Todas ~53-57% investido — o tempo flat é intrínseco ao BTC em")
        out("baixa; operar muito mais só removendo a proteção (OOS maxDD -88%). Não há almoço grátis aqui.\n")

    if quality:
        out("**REFINO DE QUALIDADE** (mesmo ~%investido, mas Pareto melhor — menos whipsaw):")
        for n, po, m, w in sorted(quality, key=lambda x: -x[2]["sharpe"]):
            out(f"- **{n}**: full Sharpe {m['sharpe']:.2f} (vs {base_m['sharpe']:.2f}), CAGR {m['ann_ret']:+.1%}, "
                f"folds {w['pos']}/{w['nf']}, OOS Sharpe {w['sharpe']:.2f}, OOS maxDD {w['maxdd']:+.0%}, {po:.0%} inv.")
        out("\n→ A histerese (band) é principiada (reduz whipsaw no cruzamento). Ganho maior é in-sample;")
        out("  OOS fica ~empatado com leve vantagem em folds. Baixo risco adotar, mas NÃO resolve 'flat'.")

    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "regime_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nRelatório salvo em {path}")


if __name__ == "__main__":
    main()
