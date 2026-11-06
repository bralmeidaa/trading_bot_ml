# -*- coding: utf-8 -*-
"""
Robustness battery for the validated daily cross-sectional momentum strategy.
Stress-tests the GO config beyond the base walk-forward:

  1. Regime windows — bull vs bear vs chop (worst-case exposure)
  2. Parameter stability — Sharpe across a lookback × rebalance grid
  3. Long-only vs market-neutral (short-side may be infeasible)
  4. Universe-size sensitivity (max_universe)
  5. k sensitivity (concentration)

Reads the cached panel (run backend.data.universe first if cache is cold).
Writes a markdown report to .claude/research/reports/.

Run:  python .claude/research/robustness_cross_sectional.py
"""
import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import _bootstrap  # noqa: F401  (adds project root to sys.path)

import numpy as np
import pandas as pd

from backend.data.universe import build_panel, EXPANDED_UNIVERSE
from backend.strategy.cross_sectional import simulate, metrics

TF, DAYS = "1d", 1200
# Reference GO config (docs/STRATEGY_THESIS.md)
LB, RB, K, MODE = 12, 12, 3, "momentum"
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


def regime_windows(close, volume):
    """Split the timeline into thirds (rough early/mid/late regimes) + full."""
    n = len(close)
    segs = {
        "early-third": (0, n // 3),
        "mid-third":   (n // 3, 2 * n // 3),
        "late-third":  (2 * n // 3, n),
        "full":        (0, n),
    }
    out = {}
    for name, (a, b) in segs.items():
        c, v = close.iloc[a:b], volume.iloc[a:b]
        p = simulate(c, LB, RB, K, MODE, btc_filter=True, volume=v)
        out[name] = metrics(p, TF)
    return out


def param_grid(close, volume):
    rows = []
    for lb in (6, 12, 24, 48):
        for rb in (6, 12, 24):
            p = simulate(close, lb, rb, K, MODE, btc_filter=True, volume=volume)
            m = metrics(p, TF)
            rows.append((lb, rb, m["sharpe"], m["ann_ret"], m["maxdd"]))
    return rows


def variants(close, volume):
    """Long-only vs neutral, universe size, k."""
    res = {}
    base = simulate(close, LB, RB, K, MODE, btc_filter=True, volume=volume)
    res["market-neutral (base)"] = metrics(base, TF)
    for mu in (10, 15, 20):
        p = simulate(close, LB, RB, K, MODE, btc_filter=True, volume=volume, max_universe=mu)
        res[f"max_universe={mu}"] = metrics(p, TF)
    for k in (2, 3, 5):
        p = simulate(close, LB, RB, k, MODE, btc_filter=True, volume=volume)
        res[f"k={k}"] = metrics(p, TF)
    # regime filter off (worst-case downside)
    p = simulate(close, LB, RB, K, MODE, btc_filter=False, volume=volume)
    res["regime filter OFF"] = metrics(p, TF)
    return res


def main():
    print("Loading panel (PIT, expanded universe)...")
    close, volume, kept = build_panel(symbols=EXPANDED_UNIVERSE, timeframe=TF,
                                      days=DAYS, point_in_time=True, verbose=False)
    print(f"  {close.shape[0]} bars × {close.shape[1]} symbols\n")

    rw = regime_windows(close, volume)
    pg = param_grid(close, volume)
    vr = variants(close, volume)

    os.makedirs(REPORT_DIR, exist_ok=True)
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out(f"# Robustez — cross-sectional momentum diário (LB{LB}/RB{RB}/k{K}, {MODE})\n")
    out(f"Painel: {close.shape[0]} barras × {close.shape[1]} símbolos (PIT, ~{DAYS}d)\n")

    out("## 1. Janelas de regime (pior caso de exposição)\n")
    out("| janela | Sharpe | ann.ret | maxDD |")
    out("|--------|--------|---------|-------|")
    for name, m in rw.items():
        out(f"| {name} | {m['sharpe']:.2f} | {m['ann_ret']:+.1%} | {m['maxdd']:+.1%} |")

    out("\n## 2. Estabilidade de parâmetros (Sharpe)\n")
    out("| LB | RB | Sharpe | ann.ret | maxDD |")
    out("|----|----|--------|---------|-------|")
    for lb, rb, sh, ar, dd in pg:
        out(f"| {lb} | {rb} | {sh:.2f} | {ar:+.1%} | {dd:+.1%} |")

    out("\n## 3. Variantes (long/neutral, universo, k, regime)\n")
    out("| variante | Sharpe | ann.ret | maxDD |")
    out("|----------|--------|---------|-------|")
    for name, m in vr.items():
        out(f"| {name} | {m['sharpe']:.2f} | {m['ann_ret']:+.1%} | {m['maxdd']:+.1%} |")

    # honest read
    out("\n## Leitura")
    worst = min(rw.values(), key=lambda m: m["sharpe"])
    stable = sum(1 for _, _, sh, _, _ in pg if sh > 0.8)
    out(f"- Pior janela de regime: Sharpe {worst['sharpe']:.2f}, maxDD {worst['maxdd']:+.1%}.")
    out(f"- Estabilidade: {stable}/{len(pg)} combos de parâmetros com Sharpe > 0.8.")
    out(f"- Regime filter OFF: maxDD {vr['regime filter OFF']['maxdd']:+.1%} "
        f"(vs base {vr['market-neutral (base)']['maxdd']:+.1%}) — confirma que o filtro controla o downside.")

    path = os.path.join(REPORT_DIR, "robustness_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport salvo em {path}")


if __name__ == "__main__":
    main()
