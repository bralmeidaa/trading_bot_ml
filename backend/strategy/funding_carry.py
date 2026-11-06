"""
Funding-rate carry (delta-neutral) — pure, testable core. Trilha B da pesquisa
de 2026-10-27 (ver plano arquivado nessa sessão e docs/STRATEGY_THESIS.md).

Mecanismo DIFERENTE de tudo que já foi testado neste projeto: não é predição de
preço. É captura estrutural do funding de perpétuos — spot LONG + perp SHORT em
tamanho igual (delta-neutral: o P&L de preço das duas pernas se cancela), e o
retorno vem só do funding pago a cada 8h. Quando funding > 0, longs pagam
shorts — este livro RECEBE. A estratégia escolhe, a cada fixing, os símbolos
com funding mais alto NO FIXING ANTERIOR (causal: o fixing atual pode não
replicar) para segurar durante o próximo período.

Risco de basis (preço spot vs. perp pode divergir um pouco, especialmente em
estresse) e custo de margem/liquidação do perp NÃO estão modelados aqui — ver
ressalvas no relatório de pesquisa.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_COST_PER_SIDE = 0.0015   # conservador: mesma convenção do cross_sectional
                                  # (taxas reais de perp/spot tendem a ser menores)
FUNDING_PERIODS_PER_YEAR = 1095  # 3 fixings/dia (8h) * 365


def select_top_m(funding_row: pd.Series, m: int, min_funding: float = 0.0) -> pd.Series:
    """
    Equal-weight the top-m symbols by funding rate, but only those with
    funding > min_funding (never force a position that would be paying, not
    receiving). Fewer than m eligible → fewer positions, weight 1/n_eligible
    each (not m) so the book stays fully carry-funded. 0 eligible → all-flat.
    """
    w = pd.Series(0.0, index=funding_row.index)
    eligible = funding_row[funding_row > min_funding].dropna()
    if eligible.empty:
        return w
    top = eligible.sort_values(ascending=False).iloc[:m]
    w[top.index] = 1.0 / len(top)
    return w


def simulate(funding: pd.DataFrame, m: int, cost_per_side: float = DEFAULT_COST_PER_SIDE,
            min_funding: float = 0.0, rebalance: int = 1) -> pd.Series:
    """
    Simulate the delta-neutral carry book. Returns per-fixing-period net return.
    No look-ahead: weights decided at fixing t use funding known AT t (just
    fixed); the payment realized for period t→t+1 uses funding fixed at t+1
    (shift — same pattern as cross_sectional.simulate). Entering/exiting a
    position costs cost_per_side on BOTH legs (spot + perp) — turnover*2*cost.

    rebalance (periods, default 1 = re-evaluate every 8h fixing): a single
    noisy fixing is a weak read on which symbol will keep paying — re-entering
    every 8h off that noise, as rebalance=1 does, lets cost×turnover dominate
    (the same mechanism that kills intraday elsewhere in this project).
    rebalance>1 holds the book between re-evaluations, same pattern as
    cross_sectional.simulate's `rebalance` — the realistic way carry bots
    actually run (re-check every few days, not every fixing).
    """
    n, syms = funding.shape
    if n < 3 or syms < 1:
        return pd.Series(dtype=float)

    weights = pd.DataFrame(0.0, index=funding.index, columns=funding.columns)
    cur_w = pd.Series(0.0, index=funding.columns)
    turnover = pd.Series(0.0, index=funding.index)

    for t in range(0, n, rebalance):
        new_w = select_top_m(funding.iloc[t], m, min_funding)
        turnover.iloc[t] = (new_w - cur_w).abs().sum()
        cur_w = new_w
        weights.iloc[t:min(t + rebalance, n)] = new_w.values

    # payment realized next period = position held (from t) * funding fixed at t+1
    carry = (weights.shift(1).fillna(0.0) * funding).sum(axis=1)
    cost = turnover * cost_per_side * 2.0   # both legs, each time the book changes
    return carry - cost


def metrics(port: pd.Series) -> dict:
    """Performance metrics from a per-fixing-period return series."""
    if port is None or len(port) == 0 or port.abs().sum() == 0:
        return dict(n=0, ann_ret=0.0, sharpe=0.0, maxdd=0.0, net=0.0)
    bpy = FUNDING_PERIODS_PER_YEAR
    mean, std = port.mean(), port.std()
    sharpe = float(mean / std * np.sqrt(bpy)) if std > 0 else 0.0
    equity = (1 + port).cumprod()
    net = float(equity.iloc[-1] - 1)
    ann = float((1 + net) ** (bpy / len(port)) - 1)
    roll_max = equity.cummax()
    maxdd = float(((equity - roll_max) / roll_max).min())
    return dict(n=len(port), ann_ret=ann, sharpe=sharpe, maxdd=maxdd, net=net)


def walk_forward(funding: pd.DataFrame, n_folds: int, m: int,
                 cost_per_side: float = DEFAULT_COST_PER_SIDE,
                 min_funding: float = 0.0, rebalance: int = 1) -> dict | None:
    """Out-of-sample walk-forward over contiguous folds; aggregate + per-fold Sharpe."""
    n = len(funding)
    fold = n // (n_folds + 1)
    sharpes, ports = [], []
    for f in range(1, n_folds + 1):
        seg = funding.iloc[f * fold:(f + 1) * fold]
        if len(seg) < 10:
            continue
        p = simulate(seg, m, cost_per_side, min_funding, rebalance)
        if len(p):
            sharpes.append(metrics(p)["sharpe"])
            ports.append(p)
    if not ports:
        return None
    agg = metrics(pd.concat(ports))
    agg["fold_sharpes"] = sharpes
    agg["pos_folds"] = sum(1 for s in sharpes if s > 0)
    agg["n_folds"] = len(sharpes)
    return agg
