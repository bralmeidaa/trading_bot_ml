# -*- coding: utf-8 -*-
"""
Walk-forward com RE-OTIMIZAÇÃO por janela curta (2026-10-27) — testa a hipótese
do usuário: "em operação curta, a janela de observação é menor, o zoom é
maior". Até aqui todo teste de robustez neste projeto escolheu UM parâmetro
fixo (LB/RB/k/mode) e cobrava dele sobreviver a 2 anos de regimes diferentes
— se ele falha, isso só prova que aquele parâmetro ESPECÍFICO não é eterno,
não prova que um operador de curto prazo (que re-calibra com frequência) não
teria edge.

Metodologia: a cada janela de treino (ex. 90 dias), escolhe o melhor config
(LB,RB,k,mode) SÓ com dados daquela janela (sem olhar o futuro), aplica esse
config OOS na janela de teste seguinte (ex. 30 dias), rola pra frente. Reporta
a série OOS inteira (concatenada) e o quão ESTÁVEL é a escolha de parâmetro
entre janelas (se mudar sem padrão a cada rolagem, é sinal de que o "melhor"
de cada janela é ruído, não edge real recalibrável).

Controle anti-ilusão: compara contra um baseline "congelado" (mesmo config
fixo em todas as janelas, sem re-otimizar) para isolar se a re-otimização
está de fato ajudando ou só pintando o viés de seleção de outra forma.

Run:
  python research_rolling_wfo.py --timeframe 1h --days 730 --train-days 90 --test-days 30
  python research_rolling_wfo.py --timeframe 4h --days 900 --train-days 120 --test-days 30
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
from collections import Counter

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from backend.data.universe import build_panel, DEFAULT_UNIVERSE
from backend.strategy.cross_sectional import simulate, metrics

COST_PER_SIDE = 0.0015

GRID_LB = {"1h": [6, 12, 24, 48], "4h": [12, 24, 48], "1d": [6, 12, 24]}
GRID_RB = {"1h": [6, 12, 24], "4h": [6, 12, 24], "1d": [6, 12]}
GRID_K = [3, 4]
GRID_MODE = ["momentum", "reversal"]


def bars_per_day(timeframe: str) -> int:
    return {"1h": 24, "4h": 6, "1d": 1}.get(timeframe, 24)


def best_insample_config(train: pd.DataFrame, volume_tr: pd.DataFrame, timeframe: str):
    """Sweep the grid on the TRAINING slice only; return the best (lb,rb,k,mode)
    by in-sample Sharpe (net of cost) and that run's in-sample Sharpe."""
    best, best_sharpe = None, -1e9
    for lb in GRID_LB.get(timeframe, [12, 24, 48]):
        for rb in GRID_RB.get(timeframe, [6, 12, 24]):
            for k in GRID_K:
                for mode in GRID_MODE:
                    port = simulate(train, lb, rb, k, mode=mode, btc_filter=True,
                                    cost=COST_PER_SIDE, volume=volume_tr)
                    if len(port) == 0 or port.abs().sum() == 0:
                        continue
                    sh = metrics(port, timeframe)["sharpe"]
                    if sh > best_sharpe:
                        best_sharpe, best = sh, (lb, rb, k, mode)
    return best, best_sharpe


def rolling_wfo(close, volume, timeframe, train_bars, test_bars, max_lookback):
    n = len(close)
    oos_returns, window_log = [], []
    start = 0
    while start + train_bars + test_bars <= n:
        train = close.iloc[start:start + train_bars]
        vtrain = volume.iloc[start:start + train_bars]
        cfg, in_sh = best_insample_config(train, vtrain, timeframe)
        test_start = start + train_bars
        test_end = test_start + test_bars
        # include max_lookback bars of warm-up BEFORE the test window so the
        # signal is computable from the test window's first bar (standard
        # practice — those bars are past data known at test_start, not future).
        warm_start = max(0, test_start - max_lookback)
        ext = close.iloc[warm_start:test_end]
        vext = volume.iloc[warm_start:test_end]
        if cfg is None:
            start += test_bars
            continue
        lb, rb, k, mode = cfg
        port_ext = simulate(ext, lb, rb, k, mode=mode, btc_filter=True,
                            cost=COST_PER_SIDE, volume=vext)
        # keep only the bars that fall within the actual test window
        port_test = port_ext.iloc[-(test_end - test_start):] if len(port_ext) else port_ext
        oos_sh = metrics(port_test, timeframe)["sharpe"] if len(port_test) else 0.0
        oos_returns.append(port_test)
        window_log.append({"cfg": cfg, "in_sample_sharpe": in_sh, "oos_sharpe": oos_sh,
                           "oos_sum_pct": float(port_test.sum() * 100) if len(port_test) else 0.0})
        start += test_bars
    return oos_returns, window_log


def frozen_baseline(close, volume, timeframe, train_bars, test_bars, max_lookback, frozen_cfg):
    """Same rolling test windows, but ALWAYS use one fixed config (no refit) —
    isolates whether re-optimization itself adds anything over a static rule."""
    n = len(close)
    oos_returns = []
    start = 0
    lb, rb, k, mode = frozen_cfg
    while start + train_bars + test_bars <= n:
        test_start = start + train_bars
        test_end = test_start + test_bars
        warm_start = max(0, test_start - max_lookback)
        ext = close.iloc[warm_start:test_end]
        vext = volume.iloc[warm_start:test_end]
        port_ext = simulate(ext, lb, rb, k, mode=mode, btc_filter=True,
                            cost=COST_PER_SIDE, volume=vext)
        port_test = port_ext.iloc[-(test_end - test_start):] if len(port_ext) else port_ext
        oos_returns.append(port_test)
        start += test_bars
    return oos_returns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--train-days", type=int, default=90)
    ap.add_argument("--test-days", type=int, default=30)
    args = ap.parse_args()

    bpd = bars_per_day(args.timeframe)
    train_bars, test_bars = args.train_days * bpd, args.test_days * bpd
    max_lb = max(GRID_LB.get(args.timeframe, [48])) * (1 if args.timeframe == "1d" else 1)

    print("=" * 78)
    print(f"ROLLING WFO — re-otimiza a cada {args.train_days}d, testa OOS nos {args.test_days}d "
          f"seguintes | {args.timeframe}, {args.days}d totais")
    print("=" * 78)

    close, volume, kept = build_panel(symbols=DEFAULT_UNIVERSE, timeframe=args.timeframe,
                                      days=args.days)

    oos_returns, log = rolling_wfo(close, volume, args.timeframe, train_bars, test_bars, max_lb)
    if not oos_returns:
        print("Poucas janelas — aumente --days ou reduza --train-days/--test-days.")
        return

    print(f"\n{len(log)} janelas de {args.test_days}d OOS:")
    print(f"{'janela':>6}{'config (LB,RB,k,modo)':>28}{'in_sh':>8}{'oos_sh':>8}{'oos_%':>8}")
    print("-" * 78)
    for i, w in enumerate(log):
        print(f"{i:>6}{str(w['cfg']):>28}{w['in_sample_sharpe']:>8.2f}"
              f"{w['oos_sharpe']:>8.2f}{w['oos_sum_pct']:>+7.2f}%")

    all_oos = pd.concat(oos_returns)
    agg = metrics(all_oos, args.timeframe)
    pos_windows = sum(1 for w in log if w["oos_sharpe"] > 0)
    print("\n" + "-" * 78)
    print(f"WFO (re-otimizado): Sharpe agregado={agg['sharpe']:.2f}  annRet={agg['ann_ret']:+.1%}  "
          f"maxDD={agg['maxdd']:+.1%}  janelas+={pos_windows}/{len(log)}")

    # estabilidade da escolha de parametro: se muda a toda hora sem padrao, é ruido
    cfgs = Counter(w["cfg"] for w in log)
    print(f"\nEstabilidade da escolha de config entre janelas (top 5 mais escolhidos):")
    for cfg, n_times in cfgs.most_common(5):
        print(f"  {cfg}: escolhido em {n_times}/{len(log)} janelas")

    # baseline congelado: pega o config mais escolhido pelo WFO e roda fixo em TODAS as janelas
    frozen_cfg = cfgs.most_common(1)[0][0]
    frozen_returns = frozen_baseline(close, volume, args.timeframe, train_bars, test_bars,
                                     max_lb, frozen_cfg)
    all_frozen = pd.concat(frozen_returns)
    agg_frozen = metrics(all_frozen, args.timeframe)
    pos_frozen = sum(1 for p in frozen_returns if len(p) and metrics(p, args.timeframe)["sharpe"] > 0)
    print(f"\nBASELINE CONGELADO ({frozen_cfg}, sem re-otimizar): Sharpe={agg_frozen['sharpe']:.2f}  "
          f"annRet={agg_frozen['ann_ret']:+.1%}  maxDD={agg_frozen['maxdd']:+.1%}  "
          f"janelas+={pos_frozen}/{len(frozen_returns)}")

    print("\n" + "=" * 78)
    go = agg["sharpe"] > 0.8 and pos_windows >= len(log) * 0.7 and agg["ann_ret"] > 0
    beats_frozen = agg["sharpe"] > agg_frozen["sharpe"] + 0.3
    print(f"VEREDITO: {'GO' if go else 'NO-GO'} para o WFO re-otimizado "
          f"(Sharpe={agg['sharpe']:.2f}, {pos_windows}/{len(log)} janelas+).")
    if go and not beats_frozen:
        print("ATENÇÃO: o WFO NÃO superou claramente o baseline congelado — a re-otimização")
        print("pode não estar adicionando edge real, só reagindo a ruído de cada janela.")
    elif go and beats_frozen:
        print("O WFO re-otimizado supera claramente o baseline congelado — re-calibrar com")
        print("frequência parece genuinamente ajudar aqui, não só reaplicar o mesmo parâmetro.")
    print("=" * 78)


if __name__ == "__main__":
    main()
