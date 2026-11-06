# -*- coding: utf-8 -*-
"""
Lead-lag event-driven (2026-10-27) — mecanismo DIFERENTE de tudo testado antes:
não é ranking cross-sectional nem previsão contínua. Dispara só quando o BTC dá
um choque forte (1h) e aposta que altcoins de alto beta reagem com atraso —
baixa frequência de trade POR CONSTRUÇÃO (ataca o cost×turnover pela raiz, ao
invés de tentar reduzir custo/turnover de uma estratégia contínua).

Veredito (ver relatório): aparenta positivo na média em thresholds extremos,
mas é artefato de cauda gorda (outlier), não edge repetível — mediana negativa,
decai pra negativo na 2a metade da janela, inconsistente entre símbolos.

Run:
  python research_leadlag.py --days 730 --timeframe 1h
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from backend.data.universe import build_panel, DEFAULT_UNIVERSE

COST_PER_SIDE = 0.0015   # custo conservador padrao do projeto (spot taker-like)


def leadlag_trades(close: pd.DataFrame, btc_col: str, threshold_q: float,
                   hold_bars: int, cooldown_bars: int = 8) -> pd.DataFrame:
    """
    Trigger: |retorno do BTC em 1 barra| >= quantil `threshold_q` da série.
    Trade: entra em TODAS as alts na direção do choque, sai em `hold_bars`.
    `cooldown_bars`: ignora novos triggers muito próximos do último (evita
    contar o mesmo evento de choque várias vezes).
    """
    btc_shock = close[btc_col].pct_change(1)
    alts = [c for c in close.columns if c != btc_col]
    thr = btc_shock.abs().quantile(threshold_q)
    triggers = btc_shock[btc_shock.abs() >= thr]
    idx_pos = {ts: i for i, ts in enumerate(close.index)}

    rows, last_i = [], -10**9
    for ts, shock in triggers.items():
        i = idx_pos[ts]
        if i - last_i < cooldown_bars:
            continue
        last_i = i
        if i + hold_bars >= len(close):
            continue
        direction = np.sign(shock)
        for sym in alts:
            entry, exitp = close[sym].iloc[i], close[sym].iloc[i + hold_bars]
            if entry > 0 and np.isfinite(entry) and np.isfinite(exitp):
                rows.append({"i": i, "ts": ts, "sym": sym,
                            "gross": direction * (exitp / entry - 1.0)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--days", type=int, default=730)
    args = ap.parse_args()

    print("=" * 78)
    print(f"LEAD-LAG — choque BTC -> reação em altcoins | {args.timeframe}, {args.days}d")
    print("=" * 78)

    close, volume, kept = build_panel(symbols=DEFAULT_UNIVERSE, timeframe=args.timeframe,
                                      days=args.days)

    print(f"\n{'thr_q':>7}{'hold':>6} | {'n':>6}{'net_mean':>10}{'net_med':>10}{'WR':>6}"
          f"{'1a_metade':>12}{'2a_metade':>12}")
    print("-" * 78)
    for thr_q in (0.90, 0.95, 0.98, 0.99, 0.995):
        for hold in (2, 4, 8, 12):
            df = leadlag_trades(close, "BTC/USDT", thr_q, hold)
            if len(df) < 40:
                continue
            net = df["gross"] - 2 * COST_PER_SIDE
            mid = df["i"].median()
            first, second = net[df["i"] <= mid], net[df["i"] > mid]
            print(f"{thr_q:>7.3f}{hold:>5}h | {len(df):>6}{net.mean()*100:>+9.3f}%"
                  f"{net.median()*100:>+9.3f}%{(net > 0).mean():>6.0%}"
                  f"{first.mean()*100:>+11.3f}%{second.mean()*100:>+11.3f}%")

    print("\n" + "=" * 78)
    print("VEREDITO: NO-GO. Nos thresholds mais extremos (q>=0.99) a MÉDIA fica")
    print("positiva, mas a MEDIANA é negativa (cauda gorda — poucos trades grandes")
    print("carregam a média) e o efeito decai para NEGATIVO na 2a metade da janela.")
    print("Quebra de robustez igual à já vista neste projeto: media positiva isolada,")
    print("sem consistência temporal nem entre símbolos, não é edge repetível.")
    print("=" * 78)


if __name__ == "__main__":
    main()
