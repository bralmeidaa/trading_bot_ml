# -*- coding: utf-8 -*-
"""
Trilha B (2026-10-27) — funding-rate carry (delta-neutral) como motor de
"operação" diferente da predição de preço: spot LONG + perp SHORT em tamanho
igual, captura o funding pago a cada 8h (long paga short quando funding>0).

Não compete com a busca por intraday/tempo-real (é um carry trade, contínuo),
mas é um mecanismo genuinamente novo neste projeto — nunca testado como
estratégia (só como feature de ML em research_features.py, que é outra coisa).

Reaproveita backend/data/universe.DEFAULT_UNIVERSE (point-in-time não é
necessário aqui — funding não tem o mesmo viés de sobrevivência de preço) e
backend/data/derivatives_features.fetch_funding_panel.

Run:
  python research_funding_carry.py --days 1200 --folds 5
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse

import ccxt
import pandas as pd

import _bootstrap  # noqa: F401
from backend.data.universe import DEFAULT_UNIVERSE
from backend.data.derivatives_features import fetch_funding_panel
from backend.strategy.funding_carry import walk_forward

COST_PER_SIDE = 0.0015   # conservador (perp taker real costuma ser ~0.04-0.05%)


def build_funding_matrix(funding_by_symbol: dict) -> pd.DataFrame:
    """
    Wide funding panel on the UNION of fixing timestamps (funding natively
    ticks every 8h — no need to warp into an OHLCV bar index here). Forward-
    fill only within each symbol's own series (handles any missed fixing);
    no fill across symbols (absence stays NaN = not eligible that period).
    """
    cols = {}
    for sym, fr in funding_by_symbol.items():
        cols[sym] = fr.set_index("timestamp")["funding"]
    if not cols:
        return pd.DataFrame()
    wide = pd.DataFrame(cols).sort_index()
    return wide.ffill(limit=2)   # tolerate at most 1 missed fixing (16h) per symbol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1200)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    print("=" * 78)
    print(f"TRILHA B — funding-rate carry (delta-neutral) — {args.days}d, {args.folds} folds")
    print(f"  custo/lado={COST_PER_SIDE*100:.2f}% (2 pernas: spot + perp)")
    print("=" * 78)

    fut = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    funding_by_symbol = fetch_funding_panel(fut, DEFAULT_UNIVERSE, args.days)
    if len(funding_by_symbol) < 4:
        print("Poucos simbolos com funding disponivel — abortando.")
        return

    funding = build_funding_matrix(funding_by_symbol)
    print(f"\nMatriz de funding: {funding.shape[0]} fixings x {funding.shape[1]} simbolos "
          f"({funding.shape[0] / 3:.0f} dias)")
    pct_positive = (funding > 0).mean().mean()
    print(f"Fracao de fixings com funding>0 (pagaria o short): {pct_positive:.1%}")

    ms = [1, 2, 3, 5]
    min_fundings = [0.0, 0.0001, 0.0003]
    rebalances = [1, 3, 9, 21, 63]   # 8h, 1d, 3d, 1wk, 3wk (periods are 8h fixings)
    rows = []
    for m in ms:
        for mf in min_fundings:
            for rb in rebalances:
                agg = walk_forward(funding, args.folds, m, COST_PER_SIDE, mf, rb)
                if agg:
                    rows.append((m, mf, rb, agg))
    rows.sort(key=lambda r: r[3]["sharpe"], reverse=True)

    print(f"\n{'m':>3}{'min_funding':>13}{'rebal':>7} | "
          f"{'Sharpe':>7}{'annRet':>9}{'maxDD':>8}{'pos/folds':>10}")
    print("-" * 78)
    for m, mf, rb, a in rows[:20]:
        print(f"{m:>3}{mf:>13.4f}{rb:>7} | {a['sharpe']:>7.2f}{a['ann_ret']:>+8.1%}{a['maxdd']:>+8.1%}"
              f"{a['pos_folds']:>5}/{a['n_folds']:<4}")

    # cost sensitivity on the best config found
    print("\n" + "-" * 78)
    print("Sensibilidade a custo (melhor config):")
    if rows:
        m, mf, rb, _ = rows[0]
        for cost in (0.0010, 0.0015, 0.0025, 0.0040):
            agg = walk_forward(funding, args.folds, m, cost, mf, rb)
            if agg:
                print(f"  custo/lado {cost*100:.2f}%: Sharpe {agg['sharpe']:.2f}  "
                      f"annRet {agg['ann_ret']:+.1%}  folds+ {agg['pos_folds']}/{agg['n_folds']}")

    print("\n" + "=" * 78)
    if rows:
        m, mf, rb, a = rows[0]
        go = a["sharpe"] > 0.8 and a["pos_folds"] >= a["n_folds"] - 1 and a["ann_ret"] > 0
        print(f"MELHOR: m={m}, min_funding={mf}, rebalance={rb} periodos ({rb*8}h)")
        print(f"  Sharpe={a['sharpe']:.2f}  annRet={a['ann_ret']:+.1%}  "
              f"maxDD={a['maxdd']:+.1%}  folds+={a['pos_folds']}/{a['n_folds']}")
        print(f"\nVEREDITO: {'GO' if go else 'NO-GO'} — " +
              ("carry de funding sustenta edge OOS, vale levar a paper." if go else
               "carry de funding nao sustenta edge OOS robusto com custo realista."))
    print("=" * 78)


if __name__ == "__main__":
    main()
