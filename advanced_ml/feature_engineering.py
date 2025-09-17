#!/usr/bin/env python3
"""
Advanced Feature Engineering System
Gera 50+ features técnicas, fundamentais e de mercado para ML
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Sistema avançado de feature engineering para trading."""
    
    def __init__(self):
        self.feature_groups = {
            'technical': [],
            'price_action': [],
            'volume': [],
            'volatility': [],
            'momentum': [],
            'trend': [],
            'market_structure': [],
            'statistical': [],
            'time_based': [],
            'cross_asset': []
        }
        
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
    def generate_all_features(self, price_data: pd.DataFrame, 
                            market_data: Dict = None) -> pd.DataFrame:
        """Gera todas as features disponíveis."""
        
        features_df = price_data.copy()
        
        # 1. Technical Indicators (15 features)
        technical_features = self._generate_technical_features(price_data)
        features_df = pd.concat([features_df, technical_features], axis=1)
        
        # 2. Price Action Features (12 features)
        price_action_features = self._generate_price_action_features(price_data)
        features_df = pd.concat([features_df, price_action_features], axis=1)
        
        # 3. Volume Features (8 features)
        volume_features = self._generate_volume_features(price_data)
        features_df = pd.concat([features_df, volume_features], axis=1)
        
        # 4. Volatility Features (10 features)
        volatility_features = self._generate_volatility_features(price_data)
        features_df = pd.concat([features_df, volatility_features], axis=1)
        
        # 5. Momentum Features (8 features)
        momentum_features = self._generate_momentum_features(price_data)
        features_df = pd.concat([features_df, momentum_features], axis=1)
        
        # 6. Trend Features (7 features)
        trend_features = self._generate_trend_features(price_data)
        features_df = pd.concat([features_df, trend_features], axis=1)
        
        # 7. Market Structure Features (6 features)
        structure_features = self._generate_market_structure_features(price_data)
        features_df = pd.concat([features_df, structure_features], axis=1)
        
        # 8. Statistical Features (5 features)
        statistical_features = self._generate_statistical_features(price_data)
        features_df = pd.concat([features_df, statistical_features], axis=1)
        
        # 9. Time-based Features (4 features)
        time_features = self._generate_time_features(price_data)
        features_df = pd.concat([features_df, time_features], axis=1)
        
        # 10. Advanced Market Data Features (se disponível)
        if market_data:
            market_features = self._generate_market_data_features(market_data)
            features_df = pd.concat([features_df, market_features], axis=1)
        
        # Remove NaN values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        return features_df
    
    def _generate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera features de indicadores técnicos."""
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        features = pd.DataFrame(index=df.index)
        
        # Moving Averages
        features['sma_10'] = talib.SMA(close, timeperiod=10)
        features['sma_20'] = talib.SMA(close, timeperiod=20)
        features['sma_50'] = talib.SMA(close, timeperiod=50)
        features['ema_12'] = talib.EMA(close, timeperiod=12)
        features['ema_26'] = talib.EMA(close, timeperiod=26)
        
        # Oscillators
        features['rsi_14'] = talib.RSI(close, timeperiod=14)
        features['rsi_7'] = talib.RSI(close, timeperiod=7)
        features['stoch_k'], features['stoch_d'] = talib.STOCH(high, low, close)
        features['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # MACD
        features['macd'], features['macd_signal'], features['macd_hist'] = talib.MACD(close)
        
        # Bollinger Bands
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = talib.BBANDS(close)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        self.feature_groups['technical'] = list(features.columns)
        return features
    
    def _generate_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera features de price action."""
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        
        features = pd.DataFrame(index=df.index)
        
        # Candle patterns
        features['doji'] = talib.CDLDOJI(open_price, high, low, close)
        features['hammer'] = talib.CDLHAMMER(open_price, high, low, close)
        features['engulfing'] = talib.CDLENGULFING(open_price, high, low, close)
        
        # Price relationships
        features['hl_ratio'] = (high - low) / close
        features['oc_ratio'] = (close - open_price) / open_price
        features['body_size'] = abs(close - open_price) / close
        features['upper_shadow'] = (high - np.maximum(close, open_price)) / close
        features['lower_shadow'] = (np.minimum(close, open_price) - low) / close
        
        # Gap analysis
        features['gap_up'] = (open_price - np.roll(close, 1)) / np.roll(close, 1)
        features['gap_filled'] = ((low <= np.roll(close, 1)) & (features['gap_up'] > 0)).astype(int)
        
        # Price momentum
        features['price_change_1'] = (close - np.roll(close, 1)) / np.roll(close, 1)
        features['price_change_5'] = (close - np.roll(close, 5)) / np.roll(close, 5)
        features['price_acceleration'] = features['price_change_1'] - np.roll(features['price_change_1'], 1)
        
        self.feature_groups['price_action'] = list(features.columns)
        return features
    
    def _generate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera features de volume."""
        
        close = df['close'].values
        volume = df['volume'].values
        
        features = pd.DataFrame(index=df.index)
        
        # Volume indicators
        features['volume_sma_10'] = talib.SMA(volume.astype(float), timeperiod=10)
        features['volume_sma_20'] = talib.SMA(volume.astype(float), timeperiod=20)
        features['volume_ratio'] = volume / features['volume_sma_20']
        
        # On Balance Volume
        features['obv'] = talib.OBV(close, volume.astype(float))
        features['obv_sma'] = talib.SMA(features['obv'], timeperiod=10)
        
        # Volume-Price Trend
        features['vpt'] = self._calculate_vpt(close, volume)
        
        # Volume oscillator
        features['volume_oscillator'] = (features['volume_sma_10'] - features['volume_sma_20']) / features['volume_sma_20']
        
        # Volume-weighted features
        features['vwap'] = self._calculate_vwap(df)
        
        self.feature_groups['volume'] = list(features.columns)
        return features
    
    def _generate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera features de volatilidade."""
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        features = pd.DataFrame(index=df.index)
        
        # Average True Range
        features['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
        features['atr_7'] = talib.ATR(high, low, close, timeperiod=7)
        features['atr_ratio'] = features['atr_7'] / features['atr_14']
        
        # Volatility measures
        features['close_volatility_10'] = pd.Series(close).rolling(10).std()
        features['close_volatility_20'] = pd.Series(close).rolling(20).std()
        features['volatility_ratio'] = features['close_volatility_10'] / features['close_volatility_20']
        
        # Range-based volatility
        features['true_range'] = talib.TRANGE(high, low, close)
        features['range_volatility'] = pd.Series(features['true_range']).rolling(14).std()
        
        # Parkinson volatility (high-low based)
        features['parkinson_vol'] = self._calculate_parkinson_volatility(high, low)
        
        # Volatility trend
        features['vol_trend'] = (features['atr_14'] - np.roll(features['atr_14'], 5)) / np.roll(features['atr_14'], 5)
        
        self.feature_groups['volatility'] = list(features.columns)
        return features
    
    def _generate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera features de momentum."""
        
        close = df['close'].values
        
        features = pd.DataFrame(index=df.index)
        
        # Rate of Change
        features['roc_5'] = talib.ROC(close, timeperiod=5)
        features['roc_10'] = talib.ROC(close, timeperiod=10)
        features['roc_20'] = talib.ROC(close, timeperiod=20)
        
        # Momentum
        features['momentum_10'] = talib.MOM(close, timeperiod=10)
        features['momentum_20'] = talib.MOM(close, timeperiod=20)
        
        # Commodity Channel Index
        features['cci'] = talib.CCI(df['high'].values, df['low'].values, close, timeperiod=14)
        
        # Ultimate Oscillator
        features['ultimate_osc'] = talib.ULTOSC(df['high'].values, df['low'].values, close)
        
        # Momentum acceleration
        features['momentum_acceleration'] = features['roc_5'] - features['roc_10']
        
        self.feature_groups['momentum'] = list(features.columns)
        return features
    
    def _generate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera features de tendência."""
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        features = pd.DataFrame(index=df.index)
        
        # ADX (Average Directional Index)
        features['adx'] = talib.ADX(high, low, close, timeperiod=14)
        features['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        features['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        # Parabolic SAR
        features['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        features['sar_signal'] = (close > features['sar']).astype(int)
        
        # Trend strength
        features['trend_strength'] = self._calculate_trend_strength(close)
        
        # Linear regression slope
        features['lr_slope'] = self._calculate_linear_regression_slope(close, 14)
        
        self.feature_groups['trend'] = list(features.columns)
        return features
    
    def _generate_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera features de estrutura de mercado."""
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        features = pd.DataFrame(index=df.index)
        
        # Support and Resistance levels
        features['resistance_distance'] = self._calculate_resistance_distance(high, close)
        features['support_distance'] = self._calculate_support_distance(low, close)
        
        # Pivot points
        pivot_data = self._calculate_pivot_points(df)
        features['pivot_r1_distance'] = (pivot_data['r1'] - close) / close
        features['pivot_s1_distance'] = (close - pivot_data['s1']) / close
        
        # Market structure patterns
        features['higher_highs'] = self._count_higher_highs(high, 10)
        features['lower_lows'] = self._count_lower_lows(low, 10)
        
        self.feature_groups['market_structure'] = list(features.columns)
        return features
    
    def _generate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera features estatísticas."""
        
        close = df['close'].values
        
        features = pd.DataFrame(index=df.index)
        
        # Statistical measures
        features['skewness_10'] = pd.Series(close).rolling(10).skew()
        features['kurtosis_10'] = pd.Series(close).rolling(10).kurt()
        features['zscore_20'] = self._calculate_zscore(close, 20)
        
        # Percentile ranks
        features['percentile_rank_20'] = self._calculate_percentile_rank(close, 20)
        
        # Autocorrelation
        features['autocorr_5'] = self._calculate_autocorrelation(close, 5)
        
        self.feature_groups['statistical'] = list(features.columns)
        return features
    
    def _generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera features baseadas em tempo."""
        
        features = pd.DataFrame(index=df.index)
        
        # Assumindo que o índice é datetime
        if hasattr(df.index, 'hour'):
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            features['is_market_open'] = ((df.index.hour >= 9) & (df.index.hour <= 16)).astype(int)
        else:
            # Se não tiver datetime index, criar features dummy
            features['hour'] = 12
            features['day_of_week'] = 2
            features['is_weekend'] = 0
            features['is_market_open'] = 1
        
        self.feature_groups['time_based'] = list(features.columns)
        return features
    
    def _generate_market_data_features(self, market_data: Dict) -> pd.DataFrame:
        """Gera features de dados de mercado avançados."""
        
        features = pd.DataFrame()
        
        # Funding rate features
        if 'funding_rate' in market_data:
            features['funding_rate'] = market_data['funding_rate']
            features['funding_rate_ma'] = pd.Series(market_data['funding_rate']).rolling(24).mean()
        
        # Open Interest features
        if 'open_interest' in market_data:
            features['oi_change'] = pd.Series(market_data['open_interest']).pct_change()
            features['oi_ma'] = pd.Series(market_data['open_interest']).rolling(24).mean()
        
        # Long/Short ratio
        if 'long_short_ratio' in market_data:
            features['ls_ratio'] = market_data['long_short_ratio']
            features['ls_ratio_ma'] = pd.Series(market_data['long_short_ratio']).rolling(12).mean()
        
        self.feature_groups['cross_asset'] = list(features.columns)
        return features
    
    # Helper methods
    def _calculate_vpt(self, close, volume):
        """Calculate Volume Price Trend."""
        price_change = np.diff(close, prepend=close[0])
        vpt = np.cumsum(volume * (price_change / close))
        return vpt
    
    def _calculate_vwap(self, df):
        """Calculate Volume Weighted Average Price."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    def _calculate_parkinson_volatility(self, high, low, window=14):
        """Calculate Parkinson volatility estimator."""
        hl_ratio = np.log(high / low)
        return pd.Series(hl_ratio).rolling(window).std() * np.sqrt(252)
    
    def _calculate_trend_strength(self, close, window=14):
        """Calculate trend strength."""
        sma = talib.SMA(close, timeperiod=window)
        deviations = np.abs(close - sma)
        avg_deviation = pd.Series(deviations).rolling(window).mean()
        return 1 - (avg_deviation / sma)
    
    def _calculate_linear_regression_slope(self, close, window=14):
        """Calculate linear regression slope."""
        slopes = []
        for i in range(len(close)):
            if i < window - 1:
                slopes.append(0)
            else:
                y = close[i-window+1:i+1]
                x = np.arange(len(y))
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.append(slope)
        return np.array(slopes)
    
    def _calculate_resistance_distance(self, high, close, window=20):
        """Calculate distance to resistance level."""
        resistance = pd.Series(high).rolling(window).max()
        return (resistance - close) / close
    
    def _calculate_support_distance(self, low, close, window=20):
        """Calculate distance to support level."""
        support = pd.Series(low).rolling(window).min()
        return (close - support) / close
    
    def _calculate_pivot_points(self, df):
        """Calculate pivot points."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        
        return {'pivot': pivot, 'r1': r1, 's1': s1}
    
    def _count_higher_highs(self, high, window):
        """Count higher highs in window."""
        higher_highs = []
        for i in range(len(high)):
            if i < window:
                higher_highs.append(0)
            else:
                window_data = high[i-window:i]
                count = sum(1 for j in range(1, len(window_data)) if window_data[j] > window_data[j-1])
                higher_highs.append(count / window)
        return np.array(higher_highs)
    
    def _count_lower_lows(self, low, window):
        """Count lower lows in window."""
        lower_lows = []
        for i in range(len(low)):
            if i < window:
                lower_lows.append(0)
            else:
                window_data = low[i-window:i]
                count = sum(1 for j in range(1, len(window_data)) if window_data[j] < window_data[j-1])
                lower_lows.append(count / window)
        return np.array(lower_lows)
    
    def _calculate_zscore(self, close, window):
        """Calculate Z-score."""
        rolling_mean = pd.Series(close).rolling(window).mean()
        rolling_std = pd.Series(close).rolling(window).std()
        return (close - rolling_mean) / rolling_std
    
    def _calculate_percentile_rank(self, close, window):
        """Calculate percentile rank."""
        ranks = []
        for i in range(len(close)):
            if i < window - 1:
                ranks.append(0.5)
            else:
                window_data = close[i-window+1:i+1]
                rank = stats.percentileofscore(window_data, close[i]) / 100
                ranks.append(rank)
        return np.array(ranks)
    
    def _calculate_autocorrelation(self, close, lag):
        """Calculate autocorrelation."""
        returns = np.diff(close) / close[:-1]
        autocorr = []
        
        # Pad with one zero to match original length
        autocorr.append(0)  # First value
        
        for i in range(len(returns)):
            if i < lag + 10:  # Need minimum data
                autocorr.append(0)
            else:
                recent_returns = returns[i-lag-9:i+1]
                if len(recent_returns) > lag:
                    corr = np.corrcoef(recent_returns[:-lag], recent_returns[lag:])[0, 1]
                    autocorr.append(corr if not np.isnan(corr) else 0)
                else:
                    autocorr.append(0)
        return np.array(autocorr)
    
    def get_feature_importance_groups(self) -> Dict:
        """Retorna grupos de features para análise de importância."""
        return self.feature_groups
    
    def scale_features(self, features_df: pd.DataFrame, method='robust') -> pd.DataFrame:
        """Escala features usando método especificado."""
        
        scaler = self.scalers[method]
        
        # Separar features numéricas
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        
        # Escalar apenas features numéricas (excluir OHLCV originais)
        exclude_columns = ['open', 'high', 'low', 'close', 'volume']
        scale_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        scaled_df = features_df.copy()
        if scale_columns:
            scaled_df[scale_columns] = scaler.fit_transform(features_df[scale_columns])
        
        return scaled_df

# Test the feature engineering system
def test_feature_engineering():
    """Test the advanced feature engineering system."""
    
    # Create sample data
    periods = 200
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
    
    base_price = 50000
    returns = np.random.normal(0.0005, 0.02, periods)
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
        'close': prices,
        'volume': np.random.normal(1000, 200, periods)
    }, index=dates)
    
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer()
    
    print("🔧 TESTE DO SISTEMA DE FEATURE ENGINEERING")
    print("=" * 60)
    
    # Generate all features
    features_df = engineer.generate_all_features(df)
    
    print(f"📊 FEATURES GERADAS:")
    print(f"Total de features: {len(features_df.columns)}")
    print(f"Shape dos dados: {features_df.shape}")
    
    # Show feature groups
    feature_groups = engineer.get_feature_importance_groups()
    for group, features in feature_groups.items():
        if features:
            print(f"\n{group.upper()}: {len(features)} features")
            print(f"  {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")
    
    # Scale features
    scaled_features = engineer.scale_features(features_df)
    
    print(f"\n📈 FEATURES ESCALADAS:")
    print(f"Shape após escalonamento: {scaled_features.shape}")
    print(f"Valores NaN: {scaled_features.isnull().sum().sum()}")
    
    # Show sample of final features
    print(f"\n🎯 AMOSTRA DAS FEATURES FINAIS:")
    feature_sample = scaled_features.iloc[-1, -10:].to_dict()
    for feature, value in feature_sample.items():
        print(f"  {feature}: {value:.4f}")

if __name__ == "__main__":
    test_feature_engineering()