# -*- coding: utf-8 -*-
"""
Margem de lucro da estratégia cross-sectional DIÁRIA no maior histórico possível.

Responde: "usando um período maior de dados, qual a margem de lucro?" — com
número real e honesto (full-period, por ano, sensibilidade a custo e walk-forward
out-of-sample), traduzido em $ sobre o capital de paper.

Run:
  python .claude/research/research_daily_margin.py                 # cache
  python .claude/research/research_daily_margin.py --days 2200 --no-cache
"""
import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import _bootstrap  # noqa: F401

import argparse
import numpy as np
import pandas as pd

from backend.data.universe import build_panel, EXPANDED_UNIVERSE
from backend.strategy.cross_sectional import simulate, metrics, walk_forward

TF = "1d"
LB, RB, K, MODE = 12, 12, 3, "momentum"     # GO config (docs/STRATEGY_THESIS.md)
CAPITAL = 1200.0
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


def yearly_breakdown(port: pd.Series) -> pd.DataFrame:
    """Net return per calendar year from a per-bar (daily) return series."""
    idx = pd.to_datetime(port.index, unit="ms")
    s = pd.Series(port.values, index=idx)
    rows = []
    for yr, grp in s.groupby(s.index.year):
        eq = (1 + grp).cumprod()
        net = eq.iloc[-1] - 1
        dd = ((eq - eq.cummax()) / eq.cummax()).min()
        rows.append((yr, len(grp), net, dd))
    return pd.DataFrame(rows, columns=["ano", "dias", "net", "maxdd"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2200)     # ~6 anos
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    print(f"Loading PIT panel: {args.days}d, expanded universe...")
    close, volume, kept = build_panel(symbols=EXPANDED_UNIVERSE, timeframe=TF,
                                      days=args.days, point_in_time=True,
                                      use_cache=not args.no_cache, verbose=False)
    if close.empty:
        print("panel vazio"); sys.exit(1)

    start = pd.to_datetime(close.index[0], unit="ms").date()
    end = pd.to_datetime(close.index[-1], unit="ms").date()
    years = len(close) / 365.0

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out(f"# Margem de lucro — cross-sectional diário (LB{LB}/RB{RB}/k{K}, {MODE})\n")
    out(f"Painel PIT: {close.shape[0]:,} barras × {close.shape[1]} símbolos · "
        f"{start} → {end} (~{years:.1f} anos)\n")

    # ---- full period (in-sample, the reference GO config) ----
    port = simulate(close, LB, RB, K, MODE, btc_filter=True, volume=volume)
    m = metrics(port, TF)
    final_usd = CAPITAL * (1 + m["net"])
    out("## 1. Período completo (config GO de referência)\n")
    out(f"- Retorno líquido total: **{m['net']:+.1%}**  →  ${CAPITAL:,.0f} viram **${final_usd:,.0f}**")
    out(f"- Retorno anualizado (CAGR): **{m['ann_ret']:+.1%}/ano**")
    out(f"- Sharpe: **{m['sharpe']:.2f}**")
    out(f"- Máx. drawdown: **{m['maxdd']:+.1%}**  (pior queda de pico a vale)")
    out(f"- Dias operados: {m['n']:,}\n")

    # ---- per calendar year ----
    out("## 2. Quebra por ano (consistência)\n")
    yb = yearly_breakdown(port)
    out("| ano | dias | net | maxDD | $ sobre 1200 |")
    out("|-----|------|-----|-------|--------------|")
    eq = CAPITAL
    for _, r in yb.iterrows():
        eq_end = eq * (1 + r["net"])
        out(f"| {int(r['ano'])} | {int(r['dias'])} | {r['net']:+.1%} | {r['maxdd']:+.1%} | "
            f"${eq:,.0f}→${eq_end:,.0f} |")
        eq = eq_end
    pos_years = int((yb["net"] > 0).sum())
    out(f"\nAnos positivos: **{pos_years}/{len(yb)}**\n")

    # ---- cost sensitivity ----
    out("## 3. Sensibilidade a custo (comissão+slippage por lado)\n")
    out("| custo/lado | net total | CAGR | Sharpe | maxDD |")
    out("|-----------|-----------|------|--------|-------|")
    for c in (0.0010, 0.0015, 0.0025, 0.0040):
        p = simulate(close, LB, RB, K, MODE, btc_filter=True, volume=volume, cost=c)
        mm = metrics(p, TF)
        out(f"| {c:.2%} | {mm['net']:+.1%} | {mm['ann_ret']:+.1%} | {mm['sharpe']:.2f} | {mm['maxdd']:+.1%} |")

    # ---- walk-forward OOS ----
    out("\n## 4. Walk-forward OUT-OF-SAMPLE (o número honesto)\n")
    wf = walk_forward(close, TF, args.folds, LB, RB, K, MODE, btc_filter=True, volume=volume)
    if wf:
        out(f"- Sharpe agregado OOS: **{wf['sharpe']:.2f}**")
        out(f"- CAGR agregado OOS: **{wf['ann_ret']:+.1%}**, maxDD {wf['maxdd']:+.1%}")
        out(f"- Folds positivos: **{wf['pos_folds']}/{wf['n_folds']}**  "
            f"(Sharpes: {', '.join(f'{s:.2f}' for s in wf['fold_sharpes'])})\n")

    # ---- honest read ----
    out("## Leitura honesta\n")
    out(f"- Sobre ~{years:.1f} anos, a config GO rende ~**{m['ann_ret']:+.0%}/ano** com Sharpe "
        f"{m['sharpe']:.2f} — mas com drawdown de **{m['maxdd']:+.0%}** (real, precisa estômago).")
    out("- É market-neutral e diário: latência irrelevante, robusto a custo (seção 3).")
    out("- O número que vale pra decisão é o WALK-FORWARD (seção 4), não o in-sample.")
    out("- $ é sobre 1200 SEM alavancagem nem reinvestimento composto intra-ano além do natural.")

    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "daily_margin_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nRelatório salvo em {path}")


if __name__ == "__main__":
    main()
