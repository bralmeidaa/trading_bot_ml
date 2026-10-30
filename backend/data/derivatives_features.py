"""
Derivatives / cross-asset feature layer — Trilha A da pesquisa de 2026-10-27
(ver docs/STRATEGY_THESIS.md e o plano arquivado na sessão daquele dia).

Funding rate como FEATURE de ML single-pair já foi testado e tem lift ~0
(research_features.py). Isto é diferente: aqui funding/OI/dominância alimentam
o RANKING cross-sectional (o único mecanismo com edge real neste projeto), via
o hook `extra_signal` de backend/strategy/cross_sectional.simulate.

Separação igual a backend/data/microstructure.py: I/O (fetch_*) isolado das
funções puras (*_panel, *_zscore) para que a lógica seja testável sem rede.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# I/O — fetch (network). Mirrors backend/data/universe.fetch_symbol_ohlcv and
# research_features.fetch_funding, generalized to a symbol list.
# ---------------------------------------------------------------------------

def fetch_funding_history(futures_exchange, symbol: str, days: int,
                          sleep: float = 0.25) -> pd.DataFrame:
    """
    Funding rate history (Binance USD-M perp, every 8h) for one symbol.
    `futures_exchange` must be a ccxt.binance instance with
    options={"defaultType": "future"}. Returns df[timestamp, funding], empty
    on failure/no data (perp may not exist for every spot symbol, e.g. no
    USDT-M listing yet).
    """
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    rows: list = []
    while True:
        try:
            batch = futures_exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
        except Exception:
            break
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1]["timestamp"] + 1
        if len(batch) < 1000:
            break
        time.sleep(sleep)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding"])
    fr = pd.DataFrame([{"timestamp": r["timestamp"], "funding": r["fundingRate"]} for r in rows])
    return fr.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_funding_panel(futures_exchange, symbols: List[str], days: int,
                        verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Funding history per symbol. Skips symbols with no perp (logged, not fatal)."""
    out = {}
    for sym in symbols:
        if verbose:
            print(f"  funding {sym}...", end="", flush=True)
        fr = fetch_funding_history(futures_exchange, sym, days)
        if fr.empty:
            if verbose:
                print(" none (sem perp ou sem historico)")
            continue
        out[sym] = fr
        if verbose:
            span_days = (fr["timestamp"].iloc[-1] - fr["timestamp"].iloc[0]) / 86_400_000
            print(f" {len(fr)} pontos, {span_days:.0f}d de cobertura")
    return out


def fetch_open_interest_coverage(futures_exchange, symbol: str,
                                 period: str = "1d", limit: int = 500) -> dict:
    """
    PROBE, não um fetch para produção: a API de OI histórico da Binance só
    retorna uma janela curta e recente (documentado ~30 dias para a maioria
    dos períodos) — bem diferente da funding rate, que tem anos de histórico.
    Esta função existe para MEDIR essa janela antes de decidir se OI entra no
    backtest de walk-forward (que precisa de ~1-2 anos) ou só como filtro ao
    vivo (sem validação histórica honesta).
    Retorna {available: bool, n: int, span_days: float, error: str|None}.
    """
    try:
        batch = futures_exchange.fetch_open_interest_history(symbol, period, limit=limit)
    except Exception as exc:
        return {"available": False, "n": 0, "span_days": 0.0, "error": str(exc)}
    if not batch:
        return {"available": False, "n": 0, "span_days": 0.0, "error": "empty"}
    span_ms = batch[-1]["timestamp"] - batch[0]["timestamp"]
    return {"available": True, "n": len(batch), "span_days": span_ms / 86_400_000, "error": None}


# ---------------------------------------------------------------------------
# Pure transforms — no I/O, testable with synthetic data.
# ---------------------------------------------------------------------------

def align_funding_to_panel(funding_by_symbol: Dict[str, pd.DataFrame],
                           panel_index: pd.Index) -> pd.DataFrame:
    """
    Build a wide funding-rate panel (index=panel_index, columns=symbols) from
    per-symbol funding history, point-in-time correct: for each panel bar at
    time t, the funding value is the LAST known funding fixing at or before t
    (merge_asof direction="backward" — never leaks a future fixing). Forward-
    filled only after the symbol's first funding timestamp; NaN before that
    (no perp yet / not listed), matching universe.build_panel's PIT convention.
    """
    cols = {}
    idx_df = pd.DataFrame({"timestamp": panel_index})
    for sym, fr in funding_by_symbol.items():
        merged = pd.merge_asof(idx_df.sort_values("timestamp"),
                               fr[["timestamp", "funding"]].sort_values("timestamp"),
                               on="timestamp", direction="backward")
        cols[sym] = merged.set_index("timestamp")["funding"].reindex(panel_index)
    if not cols:
        return pd.DataFrame(index=panel_index)
    return pd.DataFrame(cols)


def cross_sectional_zscore(panel: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """
    Per-bar (row-wise) z-score across symbols — puts funding/OI on a comparable
    scale to a momentum signal before blending via extra_signal/extra_weight.
    NaN-safe (a bar with <2 non-NaN values z-scores to NaN, excluded from the
    rank automatically since target_weights drops NaN).
    """
    mean = panel.mean(axis=1)
    std = panel.std(axis=1).replace(0.0, float("nan"))
    z = panel.sub(mean, axis=0).div(std, axis=0)
    return z.clip(lower=-clip, upper=clip)


def btc_dominance_proxy(close: pd.DataFrame) -> pd.Series:
    """
    Cross-asset 'regime' proxy without a market-cap data source: BTC's trailing
    30-bar return minus the equal-weight basket's trailing 30-bar return. A
    rising value means BTC is leading (altcoins lagging) — the classic
    "alt season vs BTC season" regime read from price alone (causal, no extra
    I/O). Returns an all-NaN series if BTC/USDT isn't in the panel.
    """
    if "BTC/USDT" not in close.columns:
        return pd.Series(float("nan"), index=close.index)
    rets30 = close.pct_change(30, fill_method=None)
    basket = rets30.drop(columns=["BTC/USDT"]).mean(axis=1)
    return rets30["BTC/USDT"] - basket
