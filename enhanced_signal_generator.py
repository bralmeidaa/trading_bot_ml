#!/usr/bin/env python3
"""
Enhanced Signal Generator with More Flexible Signal Combination
This version reduces restrictive requirements and adds better debugging.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class EnhancedSignalGenerator:
    """Enhanced signal generator with more flexible signal combination."""
    
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False
        
        # Get optimized parameters
        self.params = self._get_optimized_params(symbol, timeframe)
        
        # Signal generation counters for debugging
        self.signal_stats = {
            'momentum': 0,
            'mean_reversion': 0,
            'volume': 0,
            'ml': 0,
            'combined': 0,
            'total_checks': 0
        }
    
    def _get_optimized_params(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get optimized parameters with more lenient thresholds."""
        # More lenient parameters for increased activity
        if symbol == 'LINK/USDT':
            return {
                'momentum_threshold': 0.006,  # Reduced from 0.008
                'volume_threshold': 1.5,      # Reduced from 1.8
                'rsi_oversold': 40,           # Increased from 35
                'rsi_overbought': 60,         # Reduced from 65
                'confidence_multiplier': 1.0, # Reduced from 1.2
                'ml_threshold': 0.52          # Reduced from 0.55
            }
        elif symbol == 'ADA/USDT':
            return {
                'momentum_threshold': 0.006,
                'volume_threshold': 1.5,
                'rsi_oversold': 40,
                'rsi_overbought': 60,
                'confidence_multiplier': 1.0,
                'ml_threshold': 0.52
            }
        else:
            return {
                'momentum_threshold': 0.008,
                'volume_threshold': 1.6,
                'rsi_oversold': 38,
                'rsi_overbought': 62,
                'confidence_multiplier': 1.0,
                'ml_threshold': 0.53
            }
    
    def generate_signals(self, df: pd.DataFrame) -> List:
        """Generate trading signals with enhanced logging."""
        try:
            self.signal_stats['total_checks'] += 1
            
            # Add technical indicators
            df = self._add_indicators(df)
            
            # Train/update ML model (less frequently to avoid overfitting)
            if self.signal_stats['total_checks'] % 10 == 0:  # Update every 10 checks
                self._update_model(df)
            
            # Generate signals
            signals = []
            
            if len(df) < 50:
                logger.debug(f"{self.symbol} {self.timeframe}: Insufficient data ({len(df)} bars)")
                return signals
            
            latest_row = df.iloc[-1]
            current_price = latest_row['close']
            
            # Generate different types of signals
            momentum_signal = self._check_momentum_signal(latest_row)
            mean_reversion_signal = self._check_mean_reversion_signal(latest_row)
            volume_signal = self._check_volume_signal(df.iloc[-2:])
            ml_signal = self._check_ml_signal(df.iloc[-1:]) if self.is_fitted else None
            
            # Update stats
            if momentum_signal: self.signal_stats['momentum'] += 1
            if mean_reversion_signal: self.signal_stats['mean_reversion'] += 1
            if volume_signal: self.signal_stats['volume'] += 1
            if ml_signal: self.signal_stats['ml'] += 1
            
            # Enhanced signal combination - more flexible
            combined_signal = self._combine_signals_enhanced([
                momentum_signal, mean_reversion_signal, volume_signal, ml_signal
            ])
            
            if combined_signal:
                self.signal_stats['combined'] += 1
                
                # Calculate stop loss and take profit
                atr = latest_row.get('atr', current_price * 0.02)
                
                if combined_signal['direction'] == 1:
                    stop_loss = current_price - (atr * 1.5)
                    take_profit = current_price + (atr * 2.5)
                else:
                    stop_loss = current_price + (atr * 1.5)
                    take_profit = current_price - (atr * 2.5)
                
                from production_trading_system import TradeSignal
                signal = TradeSignal(
                    symbol=self.symbol,
                    direction=combined_signal['direction'],
                    strength=combined_signal['strength'],
                    confidence=combined_signal['confidence'],
                    timestamp=int(latest_row['timestamp']),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata=combined_signal['metadata']
                )
                
                signals.append(signal)
                
                logger.info(f"🎯 Signal Generated - {self.symbol} {self.timeframe}: "
                           f"Direction: {'LONG' if signal.direction == 1 else 'SHORT'}, "
                           f"Confidence: {signal.confidence:.3f}, "
                           f"Strength: {signal.strength:.3f}")
            
            # Log signal statistics every 50 checks
            if self.signal_stats['total_checks'] % 50 == 0:
                self._log_signal_stats()
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {self.symbol}: {e}")
            return []
    
    def _combine_signals_enhanced(self, signals: List[Optional[Dict]]) -> Optional[Dict]:
        """Enhanced signal combination with more flexible requirements."""
        valid_signals = [s for s in signals if s is not None]
        
        # More flexible: allow single strong signals or multiple weak signals
        if len(valid_signals) == 0:
            return None
        
        # If we have only one signal, check if it's strong enough
        if len(valid_signals) == 1:
            signal = valid_signals[0]
            # Allow single signals if they have high confidence and strength
            if signal['confidence'] > 0.7 and signal.get('strength', 0) > 0.6:
                return {
                    'direction': signal['direction'],
                    'strength': signal['strength'] * 0.8,  # Reduce strength for single signal
                    'confidence': signal['confidence'] * 0.9,  # Reduce confidence slightly
                    'metadata': {signal['type']: signal}
                }
        
        # Weighted voting for multiple signals
        weights = {'momentum': 0.3, 'mean_reversion': 0.25, 'volume': 0.25, 'ml': 0.2}
        
        long_vote = 0.0
        short_vote = 0.0
        total_confidence = 0.0
        metadata = {}
        
        for signal in valid_signals:
            weight = weights.get(signal['type'], 0.1)
            weighted_strength = signal['strength'] * signal['confidence'] * weight
            
            if signal['direction'] == 1:
                long_vote += weighted_strength
            else:
                short_vote += weighted_strength
            
            total_confidence += signal['confidence'] * weight
            metadata[signal['type']] = signal
        
        # More lenient decision logic
        min_vote_threshold = 0.2  # Reduced from 0.3
        
        if long_vote > short_vote and long_vote > min_vote_threshold:
            return {
                'direction': 1,
                'strength': min(long_vote, 1.0),
                'confidence': min(total_confidence, 0.95),
                'metadata': metadata
            }
        elif short_vote > long_vote and short_vote > min_vote_threshold:
            return {
                'direction': -1,
                'strength': min(short_vote, 1.0),
                'confidence': min(total_confidence, 0.95),
                'metadata': metadata
            }
        
        return None
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe."""
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_8'] = df['close'].ewm(span=8).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        
        return df
    
    def _check_momentum_signal(self, row) -> Optional[Dict]:
        """Check for momentum signals with more lenient conditions."""
        params = self.params
        
        if (row['momentum_5'] > params['momentum_threshold'] and
            row['volume_ratio'] > params['volume_threshold'] and
            row['rsi'] < params['rsi_overbought']):
            
            return {
                'type': 'momentum',
                'direction': 1,
                'strength': min(abs(row['momentum_5']) * 50, 1.0),
                'confidence': 0.7 * params['confidence_multiplier']
            }
        
        elif (row['momentum_5'] < -params['momentum_threshold'] and
              row['volume_ratio'] > params['volume_threshold'] and
              row['rsi'] > params['rsi_oversold']):
            
            return {
                'type': 'momentum',
                'direction': -1,
                'strength': min(abs(row['momentum_5']) * 50, 1.0),
                'confidence': 0.7 * params['confidence_multiplier']
            }
        
        return None
    
    def _check_mean_reversion_signal(self, row) -> Optional[Dict]:
        """Check for mean reversion signals."""
        params = self.params
        
        if (row['bb_position'] < 0.2 and row['rsi'] < params['rsi_oversold']):  # More lenient
            return {
                'type': 'mean_reversion',
                'direction': 1,
                'strength': min((params['rsi_oversold'] - row['rsi']) / params['rsi_oversold'], 1.0),
                'confidence': 0.8 * params['confidence_multiplier']
            }
        
        elif (row['bb_position'] > 0.8 and row['rsi'] > params['rsi_overbought']):  # More lenient
            return {
                'type': 'mean_reversion',
                'direction': -1,
                'strength': min((row['rsi'] - params['rsi_overbought']) / (100 - params['rsi_overbought']), 1.0),
                'confidence': 0.8 * params['confidence_multiplier']
            }
        
        return None
    
    def _check_volume_signal(self, df_slice) -> Optional[Dict]:
        """Check for volume breakout signals."""
        if len(df_slice) < 2:
            return None
        
        current = df_slice.iloc[-1]
        previous = df_slice.iloc[-2]
        
        price_change = (current['close'] - previous['close']) / previous['close']
        
        if (current['volume_ratio'] > self.params['volume_threshold'] and
            abs(price_change) > 0.003):  # Reduced from 0.005
            
            direction = 1 if price_change > 0 else -1
            
            return {
                'type': 'volume',
                'direction': direction,
                'strength': min(current['volume_ratio'] / 4, 1.0),
                'confidence': min(abs(price_change) * 100, 0.9)
            }
        
        return None
    
    def _check_ml_signal(self, df_slice) -> Optional[Dict]:
        """Check for ML-based signals."""
        if not self.is_fitted or len(df_slice) == 0:
            return None
        
        try:
            # Prepare features
            features = self._prepare_features(df_slice)
            if features.empty:
                return None
            
            # Get prediction
            X_scaled = self.scaler.transform(features)
            proba = self.model.predict_proba(X_scaled)[0]
            
            # Convert to signal
            if len(proba) >= 2:
                buy_prob = proba[1] if len(proba) == 2 else proba[1]
                
                if buy_prob > self.params['ml_threshold']:
                    return {
                        'type': 'ml',
                        'direction': 1,
                        'strength': min((buy_prob - 0.5) * 2, 1.0),
                        'confidence': buy_prob
                    }
                elif buy_prob < (1 - self.params['ml_threshold']):
                    return {
                        'type': 'ml',
                        'direction': -1,
                        'strength': min((0.5 - buy_prob) * 2, 1.0),
                        'confidence': 1 - buy_prob
                    }
            
        except Exception as e:
            logger.error(f"Error in ML signal generation: {e}")
        
        return None
    
    def _update_model(self, df: pd.DataFrame):
        """Update ML model with latest data."""
        try:
            if len(df) < 100:
                return
            
            # Prepare features and labels
            features = self._prepare_features(df)
            labels = self._create_labels(df)
            
            if features.empty or labels.empty or len(features) != len(labels):
                return
            
            # Remove NaN values
            valid_idx = ~(features.isna().any(axis=1) | labels.isna())
            X = features[valid_idx]
            y = labels[valid_idx]
            
            if len(X) < 50 or y.sum() < 5:
                return
            
            # Train model
            if not self.is_fitted:
                self.model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
            
            # Use only recent data for training
            recent_data = min(200, len(X))
            X_recent = X.iloc[-recent_data:]
            y_recent = y.iloc[-recent_data:]
            
            X_scaled = self.scaler.fit_transform(X_recent)
            self.model.fit(X_scaled, y_recent)
            self.is_fitted = True
            
            logger.debug(f"ML model updated for {self.symbol} {self.timeframe} with {len(X_recent)} samples")
            
        except Exception as e:
            logger.error(f"Error updating ML model: {e}")
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model."""
        feature_cols = [
            'sma_20', 'ema_8', 'ema_21', 'rsi', 'bb_position',
            'atr', 'volume_ratio', 'momentum_5', 'momentum_10'
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        return df[available_features].fillna(method='ffill').fillna(0)
    
    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create labels for ML training."""
        future_returns = df['close'].shift(-2) / df['close'] - 1
        labels = np.where(future_returns > 0.006, 1, 0)  # Reduced from 0.008
        return pd.Series(labels, index=df.index)
    
    def _log_signal_stats(self):
        """Log signal generation statistics."""
        total = self.signal_stats['total_checks']
        logger.info(f"📊 Signal Stats for {self.symbol} {self.timeframe} (last {total} checks):")
        logger.info(f"   Momentum: {self.signal_stats['momentum']} ({self.signal_stats['momentum']/total*100:.1f}%)")
        logger.info(f"   Mean Reversion: {self.signal_stats['mean_reversion']} ({self.signal_stats['mean_reversion']/total*100:.1f}%)")
        logger.info(f"   Volume: {self.signal_stats['volume']} ({self.signal_stats['volume']/total*100:.1f}%)")
        logger.info(f"   ML: {self.signal_stats['ml']} ({self.signal_stats['ml']/total*100:.1f}%)")
        logger.info(f"   Combined: {self.signal_stats['combined']} ({self.signal_stats['combined']/total*100:.1f}%)")
    
    def get_signal_stats(self) -> Dict:
        """Get current signal statistics."""
        return self.signal_stats.copy()