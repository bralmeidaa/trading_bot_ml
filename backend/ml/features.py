from __future__ import annotations
import pandas as pd
import numpy as np

from ..utils.indicators import compute_indicators


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df with columns ['ts','open','high','low','close','volume']
    Output: df with indicators + engineered features, NaNs dropped.
    """
    if df.empty:
        return df.copy()

    df = compute_indicators(df)

    # Simple returns
    df["ret1"] = df["close"].pct_change(1)
    df["ret3"] = df["close"].pct_change(3)
    df["ret5"] = df["close"].pct_change(5)

    # Normalizations and band positions
    if {"BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"}.issubset(df.columns):
        bbl = df["BBL_20_2.0"]
        bbu = df["BBU_20_2.0"]
        rng = (bbu - bbl).replace(0, np.nan)
        df["bb_pos"] = (df["close"] - bbl) / rng
        df["bb_bw"] = rng / df["BBM_20_2.0"].replace(0, np.nan)
    else:
        df["bb_pos"] = np.nan
        df["bb_bw"] = np.nan

    # RSI, MACD already computed by indicators
    # Candle geometry
    df["range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["body"] = (df["close"] - df["open"]).abs() / df["close"].replace(0, np.nan)

    # Volatility proxy
    if "atr14" in df.columns:
        df["atr_pct"] = df["atr14"] / df["close"].replace(0, np.nan)
    else:
        df["atr_pct"] = np.nan

    # Rolling stabilizers
    for col in ["ret1", "ret3", "ret5", "atr_pct", "bb_bw", "bb_pos"]:
        if col in df.columns:
            df[f"{col}_m3"] = df[col].rolling(3).mean()

    # Drop initial NaNs
    df = df.dropna().reset_index(drop=True)
    return df


def build_labels(df: pd.DataFrame, horizon: int = 3, cost_bp: float = 10.0) -> pd.Series:
    """
    Binary label: future return over horizon exceeds cost threshold.
    cost_bp: basis points (10 bp = 0.1%)
    """
    if df.empty:
        return pd.Series([], dtype=float)
    future_ret = df["close"].pct_change(horizon).shift(-horizon)
    threshold = cost_bp / 10000.0
    y = (future_ret > threshold).astype(int)
    return y
