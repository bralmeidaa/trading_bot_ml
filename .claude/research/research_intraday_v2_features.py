# -*- coding: utf-8 -*-
"""
Trilha A (2026-10-27) — funding rate / dominância BTC como peso ADICIONAL do
ranking cross-sectional já validado (docs/STRATEGY_THESIS.md), tentando
sobreviver em rebalance mais curto que o 1d GO atual.

IMPORTANTE, lido ANTES de rodar: 1h já é NO-GO conhecido (Sharpe -1.04,
STRATEGY_THESIS.md) pelo mecanismo custo×turnover, não por falta de sinal —
rebalancear a cada hora multiplica o turnover e qualquer feature nova teria que
mais que DOBRAR o Sharpe bruto só para empatar com o custo, o que nenhuma
feature single-pair testada neste projeto jamais fez. Por isso este harness
testa 4h (resultado anterior: marginal, Sharpe 1.08, regime OFF) e 1d (GO,
Sharpe 1.67-1.87) — 1h é testado só como controle negativo (checagem de que a
nova feature não revive o que já está morto por economia de custo).

Open interest NÃO entra no walk-forward: a API de OI histórico da Binance só
cobre uma janela curta e recente (ver fetch_open_interest_coverage) —
insuficiente para folds OOS de anos. Medimos a janela real e reportamos, mas
não fingimos robustez que os dados não sustentam.

Run:
  python research_intraday_v2_features.py --timeframe 4h --days 700
  python research_intraday_v2_features.py --timeframe 1d --days 1200
  python research_intraday_v2_features.py --timeframe 1h --days 180   # controle
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse

import ccxt
import pandas as pd

import _bootstrap  # noqa: F401  (adds project root to sys.path)
from backend.data.universe import build_panel, DEFAULT_UNIVERSE
from backend.data.derivatives_features import (
    fetch_funding_panel, fetch_open_interest_coverage,
    align_funding_to_panel, cross_sectional_zscore, btc_dominance_proxy,
)
from backend.strategy.cross_sectional import simulate, metrics

COST_PER_SIDE = 0.0015


def probe_oi_coverage(symbols):
    """Measure how far back Binance's OI history actually goes (honest, not assumed)."""
    fut = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    print("\nProbing OI history coverage (Binance futures)...")
    rows = []
    for sym in symbols[:5]:   # a handful is enough to characterize the window
        cov = fetch_open_interest_coverage(fut, sym, period="1d", limit=500)
        rows.append((sym, cov))
        if cov["available"]:
            print(f"  {sym}: {cov['n']} pontos, {cov['span_days']:.0f}d de cobertura")
        else:
            print(f"  {sym}: indisponivel ({cov['error']})")
    return rows


def build_extra_signal(close: pd.DataFrame, funding_z: pd.DataFrame | None,
                       dom_weight: float) -> pd.DataFrame:
    """Combine funding z-score + BTC-dominance proxy into one extra_signal panel."""
    extra = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if funding_z is not None:
        extra = extra.add(funding_z.reindex_like(extra).fillna(0.0), fill_value=0.0)
    if dom_weight:
        dom = btc_dominance_proxy(close)
        # dominance is a SINGLE series (market-wide), not per-symbol: apply it
        # as a tilt that favors non-BTC when alts are leading, and vice-versa.
        tilt = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for c in close.columns:
            tilt[c] = -dom if c == "BTC/USDT" else dom
        extra = extra + dom_weight * tilt.fillna(0.0)
    return extra


def sweep(close, volume, funding_z, timeframe, folds, lookbacks, rebalances, ks,
         extra_weights):
    rows = []
    for lb in lookbacks:
        for rb in rebalances:
            for k in ks:
                for ew in extra_weights:
                    extra = build_extra_signal(close, funding_z, dom_weight=ew) if ew else None
                    port_folds = []
                    n = len(close)
                    fold_size = n // (folds + 1)
                    sharpes = []
                    for f in range(1, folds + 1):
                        seg = close.iloc[f * fold_size:(f + 1) * fold_size]
                        vseg = volume.iloc[f * fold_size:(f + 1) * fold_size]
                        eseg = extra.iloc[f * fold_size:(f + 1) * fold_size] if extra is not None else None
                        if len(seg) < lb + rb + 5:
                            continue
                        p = simulate(seg, lb, rb, k, btc_filter=True, volume=vseg,
                                    extra_signal=eseg, extra_weight=(ew if ew else 0.0))
                        if len(p):
                            sharpes.append(metrics(p, timeframe)["sharpe"])
                            port_folds.append(p)
                    if not port_folds:
                        continue
                    agg = metrics(pd.concat(port_folds), timeframe)
                    agg["fold_sharpes"] = sharpes
                    agg["pos_folds"] = sum(1 for s in sharpes if s > 0)
                    agg["n_folds"] = len(sharpes)
                    rows.append((lb, rb, k, ew, agg))
    rows.sort(key=lambda r: r[4]["sharpe"], reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--days", type=int, default=700)
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()

    print("=" * 78)
    print(f"TRILHA A — funding + dominancia BTC no ranking cross-sectional")
    print(f"  timeframe={args.timeframe}  days={args.days}  folds={args.folds}")
    print("=" * 78)

    probe_oi_coverage(DEFAULT_UNIVERSE)

    close, volume, kept = build_panel(symbols=DEFAULT_UNIVERSE, timeframe=args.timeframe,
                                      days=args.days)
    if close.shape[1] < 6:
        print("Universo insuficiente apos build_panel — abortando.")
        return

    fut = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    funding_by_symbol = fetch_funding_panel(fut, kept, args.days)
    funding_panel = align_funding_to_panel(funding_by_symbol, close.index)
    funding_z = cross_sectional_zscore(funding_panel) if not funding_panel.empty else None
    if funding_z is None:
        print("\nAVISO: sem funding disponivel para nenhum simbolo — extra_weight!=0 sera no-op.")

    lookbacks = {"1h": [12, 24, 48], "4h": [12, 24, 48], "1d": [6, 12, 24]}.get(args.timeframe, [12, 24, 48])
    rebalances = {"1h": [6, 12, 24], "4h": [6, 12, 24], "1d": [6, 12]}.get(args.timeframe, [6, 12, 24])
    ks = [2, 3]
    extra_weights = [0.0, 0.5, 1.0, 2.0]   # 0.0 = baseline (no funding/dominance) for comparison

    rows = sweep(close, volume, funding_z, args.timeframe, args.folds,
                lookbacks, rebalances, ks, extra_weights)

    print(f"\n{'LB':>4}{'RB':>4}{'k':>3}{'extra_w':>9} | "
          f"{'Sharpe':>7}{'annRet':>9}{'maxDD':>8}{'pos/folds':>10}")
    print("-" * 78)
    for lb, rb, k, ew, a in rows[:20]:
        print(f"{lb:>4}{rb:>4}{k:>3}{ew:>9.1f} | "
              f"{a['sharpe']:>7.2f}{a['ann_ret']:>+8.1%}{a['maxdd']:>+8.1%}"
              f"{a['pos_folds']:>5}/{a['n_folds']:<4}")

    baseline = [r for r in rows if r[3] == 0.0]
    with_feature = [r for r in rows if r[3] > 0.0]
    best_base = max(baseline, key=lambda r: r[4]["sharpe"]) if baseline else None
    best_feat = max(with_feature, key=lambda r: r[4]["sharpe"]) if with_feature else None

    print("\n" + "=" * 78)
    if best_base and best_feat:
        bb, bf = best_base[4]["sharpe"], best_feat[4]["sharpe"]
        lift = bf - bb
        print(f"Melhor SEM funding/dominancia: Sharpe {bb:.2f} (LB{best_base[0]}/RB{best_base[1]}/k{best_base[2]})")
        print(f"Melhor COM funding/dominancia: Sharpe {bf:.2f} (LB{best_feat[0]}/RB{best_feat[1]}/k{best_feat[2]}/w{best_feat[3]})")
        print(f"LIFT da feature nova: {lift:+.2f}")
        go = lift > 0.15 and best_feat[4]["sharpe"] > 1.0 and best_feat[4]["pos_folds"] >= best_feat[4]["n_folds"] - 1
        print(f"\nVEREDITO: {'GO' if go else 'NO-GO'} — " +
              ("funding/dominancia elevam o edge o suficiente para justificar o risco de overfitting."
               if go else
               "funding/dominancia nao elevam o edge de forma robusta neste timeframe/janela."))
    else:
        print("Sweep nao produziu resultados suficientes para comparar.")
    print("=" * 78)


if __name__ == "__main__":
    main()
