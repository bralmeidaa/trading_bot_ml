from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler

try:
    from ..utils.cache import cache_features
except ImportError:
    # Fallback if cache not available
    def cache_features(ttl=1800):
        def decorator(func):
            return func
        return decorator

try:
    from ..utils.indicators import compute_indicators
    from ..utils.advanced_indicators import (
        compute_advanced_indicators,
        compute_statistical_features,
        compute_microstructure_features,
        compute_regime_features,
        compute_time_features
    )
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.indicators import compute_indicators
    from utils.advanced_indicators import (
        compute_advanced_indicators,
        compute_statistical_features,
        compute_microstructure_features,
        compute_regime_features,
        compute_time_features
    )


@cache_features(ttl=1800)
def build_features(df: pd.DataFrame, advanced: bool = True) -> pd.DataFrame:
    """
    Input: df with columns ['ts','open','high','low','close','volume']
    Output: df with indicators + engineered features, NaNs dropped.
    """
    if df.empty:
        return df.copy()

    # Basic indicators
    df = compute_indicators(df)
    
    # Advanced indicators if requested
    if advanced:
        df = compute_advanced_indicators(df)
        df = compute_statistical_features(df, window=20)
        df = compute_microstructure_features(df)
        df = compute_regime_features(df)
        df = compute_time_features(df)

    # Simple returns
    df["ret1"] = df["close"].pct_change(1)
    df["ret3"] = df["close"].pct_change(3)
    df["ret5"] = df["close"].pct_change(5)
    df["ret10"] = df["close"].pct_change(10)

    # Log returns (more stable for ML)
    df["log_ret1"] = np.log(df["close"] / df["close"].shift(1))
    df["log_ret3"] = np.log(df["close"] / df["close"].shift(3))
    df["log_ret5"] = np.log(df["close"] / df["close"].shift(5))

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

    # Volume features
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, np.nan)
    df["price_volume"] = df["close"] * df["volume"]
    
    # Price momentum features
    for period in [5, 10, 20]:
        df[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1
        df[f"rsi_momentum_{period}"] = df["rsi14"] - df["rsi14"].shift(period)
    
    # Cross-asset features (if multiple symbols available)
    # This would be enhanced in a multi-symbol environment
    
    # Rolling stabilizers and smoothing
    for col in ["ret1", "ret3", "ret5", "atr_pct", "bb_bw", "bb_pos", "rsi14"]:
        if col in df.columns:
            df[f"{col}_m3"] = df[col].rolling(3).mean()
            df[f"{col}_m5"] = df[col].rolling(5).mean()
            df[f"{col}_ema3"] = df[col].ewm(span=3).mean()

    # Interaction features
    if "rsi14" in df.columns and "bb_pos" in df.columns:
        df["rsi_bb_interaction"] = df["rsi14"] * df["bb_pos"]
    
    if "volume_ratio" in df.columns and "ret1" in df.columns:
        df["volume_price_interaction"] = df["volume_ratio"] * abs(df["ret1"])

    # Drop initial NaNs
    df = df.dropna().reset_index(drop=True)
    return df


def select_features(X: pd.DataFrame, y: pd.Series, method: str = "mutual_info", k: int = 50) -> list:
    """
    Select top k features using specified method.
    """
    if X.empty or len(y) == 0:
        return []
    
    # Remove non-numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]
    
    if X_numeric.empty:
        return []
    
    # Handle NaN values
    X_clean = X_numeric.fillna(X_numeric.median())
    
    if method == "mutual_info":
        selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X_clean.shape[1]))
    else:  # f_regression
        selector = SelectKBest(score_func=f_regression, k=min(k, X_clean.shape[1]))
    
    try:
        selector.fit(X_clean, y)
        selected_features = X_clean.columns[selector.get_support()].tolist()
        return selected_features
    except Exception:
        # Fallback to all numeric features if selection fails
        return numeric_cols[:k]


def create_feature_interactions(df: pd.DataFrame, max_interactions: int = 10) -> pd.DataFrame:
    """
    Create polynomial and interaction features for top features.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Select key features for interactions
    key_features = ["rsi14", "bb_pos", "atr_pct", "ret1", "volume_ratio"]
    available_features = [f for f in key_features if f in df.columns]
    
    interaction_count = 0
    for i, feat1 in enumerate(available_features):
        if interaction_count >= max_interactions:
            break
        for feat2 in available_features[i+1:]:
            if interaction_count >= max_interactions:
                break
            df[f"{feat1}_{feat2}_interaction"] = df[feat1] * df[feat2]
            interaction_count += 1
    
    # Polynomial features for key indicators
    for feat in available_features[:3]:  # Limit to top 3 to avoid explosion
        if interaction_count >= max_interactions:
            break
        df[f"{feat}_squared"] = df[feat] ** 2
        interaction_count += 1
    
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
