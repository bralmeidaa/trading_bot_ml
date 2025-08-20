"""
Advanced technical indicators for enhanced feature engineering.
"""
from typing import Optional
import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy import stats


def compute_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute advanced technical indicators beyond basic ones.
    Expects columns: ['ts','open','high','low','close','volume']
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Set datetime index for pandas_ta compatibility
    if 'ts' in df.columns:
        df.index = pd.to_datetime(df['ts'], unit='ms')
    
    # Volume-based indicators (with error handling)
    try:
        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    except:
        df["vwap"] = np.nan
    
    try:
        df["ad"] = ta.ad(df["high"], df["low"], df["close"], df["volume"])  # Accumulation/Distribution
    except:
        df["ad"] = np.nan
    
    try:
        df["obv"] = ta.obv(df["close"], df["volume"])  # On Balance Volume
    except:
        df["obv"] = np.nan
    
    try:
        df["cmf"] = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)  # Chaikin Money Flow
    except:
        df["cmf"] = np.nan
    
    try:
        df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)  # Money Flow Index
    except:
        df["mfi"] = np.nan
    
    # Volatility indicators (with error handling)
    try:
        kc_result = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=2.0)
        if kc_result is not None:
            df["kc_upper"] = kc_result.get("KCUe_20_2.0", np.nan)
            df["kc_lower"] = kc_result.get("KCLe_20_2.0", np.nan)
            df["kc_middle"] = kc_result.get("KCBe_20_2.0", np.nan)
        else:
            df["kc_upper"] = df["kc_lower"] = df["kc_middle"] = np.nan
    except:
        df["kc_upper"] = df["kc_lower"] = df["kc_middle"] = np.nan
    
    # Donchian Channels
    try:
        dc_result = ta.donchian(df["high"], df["low"], upper_length=20, lower_length=20)
        if dc_result is not None:
            df["dc_upper"] = dc_result.get("DCU_20_20", np.nan)
            df["dc_lower"] = dc_result.get("DCL_20_20", np.nan)
            df["dc_middle"] = dc_result.get("DCM_20_20", np.nan)
        else:
            df["dc_upper"] = df["dc_lower"] = df["dc_middle"] = np.nan
    except:
        df["dc_upper"] = df["dc_lower"] = df["dc_middle"] = np.nan
    
    # Momentum indicators
    try:
        df["roc"] = ta.roc(df["close"], length=10)  # Rate of Change
    except:
        df["roc"] = np.nan
    
    try:
        tsi_result = ta.tsi(df["close"], fast=25, slow=13, signal=13)
        df["tsi"] = tsi_result.get("TSI_25_13_13", np.nan) if tsi_result is not None else np.nan
    except:
        df["tsi"] = np.nan
    
    try:
        df["uo"] = ta.uo(df["high"], df["low"], df["close"])  # Ultimate Oscillator
    except:
        df["uo"] = np.nan
    
    try:
        df["willr"] = ta.willr(df["high"], df["low"], df["close"], length=14)  # Williams %R
    except:
        df["willr"] = np.nan
    
    try:
        df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)  # Commodity Channel Index
    except:
        df["cci"] = np.nan
    
    # Trend indicators
    try:
        adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx"] = adx_result.get("ADX_14", np.nan) if adx_result is not None else np.nan
    except:
        df["adx"] = np.nan
    
    try:
        aroon_result = ta.aroon(df["high"], df["low"], length=14)
        if aroon_result is not None:
            df["aroon_up"] = aroon_result.get("AROONU_14", np.nan)
            df["aroon_down"] = aroon_result.get("AROOND_14", np.nan)
        else:
            df["aroon_up"] = df["aroon_down"] = np.nan
    except:
        df["aroon_up"] = df["aroon_down"] = np.nan
    
    try:
        psar_result = ta.psar(df["high"], df["low"], af0=0.02, af=0.02, max_af=0.2)
        df["psar"] = psar_result.get("PSARl_0.02_0.2", np.nan) if psar_result is not None else np.nan
    except:
        df["psar"] = np.nan
    
    # Multiple timeframe EMAs
    for period in [5, 10, 21, 50, 100, 200]:
        try:
            df[f"ema{period}"] = ta.ema(df["close"], length=period)
        except:
            df[f"ema{period}"] = np.nan
    
    # Ichimoku components (simplified)
    try:
        ichimoku = ta.ichimoku(df["high"], df["low"], df["close"])
        if ichimoku is not None and hasattr(ichimoku, 'columns') and len(ichimoku.columns) > 0:
            df["tenkan"] = ichimoku.iloc[:, 0] if len(ichimoku.columns) > 0 else np.nan
            df["kijun"] = ichimoku.iloc[:, 1] if len(ichimoku.columns) > 1 else np.nan
            df["senkou_a"] = ichimoku.iloc[:, 2] if len(ichimoku.columns) > 2 else np.nan
            df["senkou_b"] = ichimoku.iloc[:, 3] if len(ichimoku.columns) > 3 else np.nan
        else:
            df["tenkan"] = df["kijun"] = df["senkou_a"] = df["senkou_b"] = np.nan
    except:
        df["tenkan"] = df["kijun"] = df["senkou_a"] = df["senkou_b"] = np.nan
    
    # Reset index to avoid issues
    df.reset_index(drop=True, inplace=True)
    
    return df


def compute_statistical_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute statistical features from price data.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Rolling statistics
    df[f"close_mean_{window}"] = df["close"].rolling(window).mean()
    df[f"close_std_{window}"] = df["close"].rolling(window).std()
    df[f"close_skew_{window}"] = df["close"].rolling(window).skew()
    df[f"close_kurt_{window}"] = df["close"].rolling(window).kurt()
    
    # Volume statistics
    df[f"volume_mean_{window}"] = df["volume"].rolling(window).mean()
    df[f"volume_std_{window}"] = df["volume"].rolling(window).std()
    
    # Price percentiles
    df[f"close_pct25_{window}"] = df["close"].rolling(window).quantile(0.25)
    df[f"close_pct75_{window}"] = df["close"].rolling(window).quantile(0.75)
    
    # Z-scores
    mean_close = df["close"].rolling(window).mean()
    std_close = df["close"].rolling(window).std()
    df[f"close_zscore_{window}"] = (df["close"] - mean_close) / std_close.replace(0, np.nan)
    
    return df


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market microstructure features.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Price spreads and ranges
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    df["oc_spread"] = (df["open"] - df["close"]) / df["close"]
    df["hc_spread"] = (df["high"] - df["close"]) / df["close"]
    df["lc_spread"] = (df["close"] - df["low"]) / df["close"]
    
    # Candle patterns
    df["doji"] = (abs(df["close"] - df["open"]) / (df["high"] - df["low"])).replace([np.inf, -np.inf], np.nan)
    df["hammer"] = ((df["close"] - df["low"]) / (df["high"] - df["low"])).replace([np.inf, -np.inf], np.nan)
    df["shooting_star"] = ((df["high"] - df["close"]) / (df["high"] - df["low"])).replace([np.inf, -np.inf], np.nan)
    
    # Body and shadow ratios
    body = abs(df["close"] - df["open"])
    upper_shadow = df["high"] - df[["close", "open"]].max(axis=1)
    lower_shadow = df[["close", "open"]].min(axis=1) - df["low"]
    total_range = df["high"] - df["low"]
    
    df["body_ratio"] = body / total_range.replace(0, np.nan)
    df["upper_shadow_ratio"] = upper_shadow / total_range.replace(0, np.nan)
    df["lower_shadow_ratio"] = lower_shadow / total_range.replace(0, np.nan)
    
    # Gap analysis
    df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    df["gap_filled"] = ((df["low"] <= df["close"].shift(1)) & (df["gap"] > 0)) | \
                       ((df["high"] >= df["close"].shift(1)) & (df["gap"] < 0))
    
    return df


def compute_regime_features(df: pd.DataFrame, short_window: int = 10, long_window: int = 50) -> pd.DataFrame:
    """
    Compute market regime features.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Trend regime
    df["trend_short"] = df["close"].rolling(short_window).mean()
    df["trend_long"] = df["close"].rolling(long_window).mean()
    df["trend_regime"] = (df["trend_short"] > df["trend_long"]).astype(int)
    
    # Volatility regime
    df["vol_short"] = df["close"].pct_change().rolling(short_window).std()
    df["vol_long"] = df["close"].pct_change().rolling(long_window).std()
    df["vol_regime"] = (df["vol_short"] > df["vol_long"]).astype(int)
    
    # Volume regime
    df["vol_avg_short"] = df["volume"].rolling(short_window).mean()
    df["vol_avg_long"] = df["volume"].rolling(long_window).mean()
    df["volume_regime"] = (df["vol_avg_short"] > df["vol_avg_long"]).astype(int)
    
    # Market efficiency (Hurst exponent approximation)
    def hurst_approx(series, max_lag=20):
        """Approximate Hurst exponent using R/S analysis"""
        if len(series) < max_lag * 2:
            return np.nan
        
        lags = range(2, min(max_lag, len(series) // 2))
        rs_values = []
        
        for lag in lags:
            # Calculate R/S for this lag
            y = series.values
            n = len(y) // lag * lag
            y = y[:n].reshape(-1, lag)
            
            means = np.mean(y, axis=1, keepdims=True)
            y_centered = y - means
            cumsum = np.cumsum(y_centered, axis=1)
            
            R = np.max(cumsum, axis=1) - np.min(cumsum, axis=1)
            S = np.std(y, axis=1, ddof=1)
            S[S == 0] = 1e-8  # Avoid division by zero
            
            rs_values.append(np.mean(R / S))
        
        if len(rs_values) < 2:
            return np.nan
        
        # Linear regression of log(R/S) vs log(lag)
        log_lags = np.log(lags)
        log_rs = np.log(rs_values)
        
        # Remove any infinite or NaN values
        valid_mask = np.isfinite(log_lags) & np.isfinite(log_rs)
        if np.sum(valid_mask) < 2:
            return np.nan
        
        slope, _, _, _, _ = stats.linregress(log_lags[valid_mask], log_rs[valid_mask])
        return slope
    
    # Calculate Hurst exponent for rolling windows
    hurst_window = 100
    df["hurst"] = df["close"].rolling(hurst_window).apply(lambda x: hurst_approx(x), raw=False)
    
    return df


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time-based features.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Convert timestamp to datetime if not already
    if "dt" not in df.columns:
        df["dt"] = pd.to_datetime(df["ts"], unit="ms")
    
    # Time features
    df["hour"] = df["dt"].dt.hour
    df["day_of_week"] = df["dt"].dt.dayofweek
    df["day_of_month"] = df["dt"].dt.day
    df["month"] = df["dt"].dt.month
    
    # Cyclical encoding for time features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    # Market session indicators (assuming UTC timestamps)
    # Asian session: 00:00-09:00 UTC
    # European session: 07:00-16:00 UTC  
    # US session: 13:00-22:00 UTC
    df["asian_session"] = ((df["hour"] >= 0) & (df["hour"] < 9)).astype(int)
    df["european_session"] = ((df["hour"] >= 7) & (df["hour"] < 16)).astype(int)
    df["us_session"] = ((df["hour"] >= 13) & (df["hour"] < 22)).astype(int)
    df["session_overlap"] = (df["asian_session"] + df["european_session"] + df["us_session"]) > 1
    
    return df