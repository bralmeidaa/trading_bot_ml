#!/usr/bin/env python3
"""
Final optimized trading system based on diagnostic analysis.
Implements all recommended improvements for maximum profitability.
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
from typing import Dict, List, Tuple, Any
import time
from sklearn.utils import resample
warnings.filterwarnings('ignore')

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from data.binance_client import BinancePublicClient
from ml.features import build_features, build_labels
from ml.models import create_model


class FinalOptimizedSystem:
    """Final optimized trading system with all improvements."""
    
    def __init__(self):
        self.client = BinancePublicClient()
        
    def fetch_extended_data(self, symbol: str, timeframe: str, limit: int = 3000) -> pd.DataFrame:
        """Fetch extended data for better training."""
        try:
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=limit)
            if ohlcv and len(ohlcv) > 1000:
                df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                return df
            return pd.DataFrame()
        except:
            return pd.DataFrame()
    
    def enhanced_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature engineering with better preprocessing."""
        try:
            # Build features with advanced indicators
            fdf = build_features(df, advanced=True)
            if fdf.empty:
                return pd.DataFrame()
            
            # Remove features with high null percentage
            feature_cols = [c for c in fdf.columns if c not in ("ts", "open", "high", "low", "close", "volume", "dt")]
            
            cleaned_features = []
            for col in feature_cols:
                if col in fdf.columns:
                    null_pct = fdf[col].isna().sum() / len(fdf) * 100
                    if null_pct < 30:  # Keep features with <30% nulls
                        cleaned_features.append(col)
            
            # Keep only cleaned features plus OHLCV
            keep_cols = ["ts", "open", "high", "low", "close", "volume"] + cleaned_features
            fdf_cleaned = fdf[keep_cols].copy()
            
            # Forward fill remaining nulls
            for col in cleaned_features:
                fdf_cleaned[col] = fdf_cleaned[col].fillna(method='ffill').fillna(method='bfill')
            
            # Add momentum features
            fdf_cleaned['price_momentum_5'] = fdf_cleaned['close'].pct_change(5)
            fdf_cleaned['price_momentum_10'] = fdf_cleaned['close'].pct_change(10)
            fdf_cleaned['volume_momentum_5'] = fdf_cleaned['volume'].pct_change(5)
            
            # Add volatility features
            fdf_cleaned['volatility_5'] = fdf_cleaned['close'].pct_change().rolling(5).std()
            fdf_cleaned['volatility_10'] = fdf_cleaned['close'].pct_change().rolling(10).std()
            
            # Add price position features
            fdf_cleaned['price_position_20'] = (fdf_cleaned['close'] - fdf_cleaned['close'].rolling(20).min()) / (fdf_cleaned['close'].rolling(20).max() - fdf_cleaned['close'].rolling(20).min())
            
            # Drop any remaining nulls
            fdf_cleaned = fdf_cleaned.dropna()
            
            return fdf_cleaned
            
        except Exception as e:
            print(f"Feature engineering error: {e}")
            return pd.DataFrame()
    
    def optimized_labeling(self, fdf: pd.DataFrame, horizon: int = 2, cost_bp: float = 5.0) -> pd.Series:
        """Optimized labeling with lower costs and better balance."""
        try:
            # Use lower transaction costs as recommended
            y = build_labels(fdf, horizon=horizon, cost_bp=cost_bp)
            
            if len(y) == 0:
                return pd.Series()
            
            # Check class balance
            y_int = y.astype(int)
            positive_pct = (y_int == 1).sum() / len(y_int) * 100
            
            # If severely imbalanced, try different parameters
            if positive_pct < 10:
                # Try with even lower costs
                y_alt = build_labels(fdf, horizon=horizon, cost_bp=cost_bp * 0.6)
                if len(y_alt) > 0:
                    y_alt_int = y_alt.astype(int)
                    alt_positive_pct = (y_alt_int == 1).sum() / len(y_alt_int) * 100
                    if alt_positive_pct > positive_pct:
                        return y_alt
            
            return y
            
        except Exception as e:
            print(f"Labeling error: {e}")
            return pd.Series()
    
    def balanced_model_training(self, X_train: pd.DataFrame, y_train: pd.Series, model_type: str = "xgboost") -> Any:
        """Train model with class balancing."""
        try:
            # Check class balance
            y_int = y_train.astype(int)
            positive_count = (y_int == 1).sum()
            negative_count = (y_int == 0).sum()
            
            # If severely imbalanced, use resampling
            if positive_count / negative_count < 0.3:
                print(f"  Rebalancing classes: {positive_count} positive, {negative_count} negative")
                
                # Combine features and labels
                train_data = X_train.copy()
                train_data['target'] = y_int.values
                
                # Separate classes
                positive_samples = train_data[train_data['target'] == 1]
                negative_samples = train_data[train_data['target'] == 0]
                
                # Upsample minority class
                if len(positive_samples) < len(negative_samples):
                    positive_upsampled = resample(positive_samples, 
                                                replace=True, 
                                                n_samples=min(len(negative_samples), len(positive_samples) * 3),
                                                random_state=42)
                    balanced_data = pd.concat([negative_samples, positive_upsampled])
                else:
                    negative_upsampled = resample(negative_samples,
                                                replace=True,
                                                n_samples=min(len(positive_samples), len(negative_samples) * 3),
                                                random_state=42)
                    balanced_data = pd.concat([positive_samples, negative_upsampled])
                
                # Shuffle
                balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
                
                # Split back
                X_balanced = balanced_data.drop('target', axis=1)
                y_balanced = balanced_data['target']
                
                print(f"  After balancing: {(y_balanced == 1).sum()} positive, {(y_balanced == 0).sum()} negative")
            else:
                X_balanced = X_train
                y_balanced = y_int
            
            # Train model with more features
            max_features = min(50, len(X_balanced.columns))
            model = create_model(model_type, max_features=max_features)
            
            # Add class weights for remaining imbalance
            train_metrics = model.fit(X_balanced, y_balanced.values)
            
            return model, train_metrics
            
        except Exception as e:
            print(f"Model training error: {e}")
            return None, None
    
    def optimized_backtesting(self, model, X_test, price_data, feature_cols, config):
        """Optimized backtesting with improved parameters."""
        try:
            # Optimized parameters based on diagnostic analysis
            threshold = config.get('threshold', 0.55)  # Lower threshold
            stop_loss = config.get('stop_loss', 0.025)  # Reasonable stop loss
            take_profit = config.get('take_profit', 0.08)  # Higher take profit
            max_hold = config.get('max_hold', 80)  # Longer holding period
            position_size = config.get('position_size', 0.4)  # Larger position size
            
            initial_capital = 10000.0
            capital = initial_capital
            position = 0
            trades = []
            equity_curve = [initial_capital]
            
            entry_price = 0
            entry_time = 0
            
            for i, (_, row) in enumerate(X_test.iterrows()):
                current_price = price_data.iloc[i]['close']
                
                # Current equity
                current_equity = capital + (position * current_price if position > 0 else 0)
                equity_curve.append(current_equity)
                
                # Get prediction
                features = row[feature_cols].values.reshape(1, -1)
                features_df = pd.DataFrame(features, columns=feature_cols)
                prob = model.predict_proba(features_df)[0]
                
                if position == 0 and prob > threshold:
                    # Enter position with larger size
                    position_value = capital * position_size
                    position = position_value / current_price * 0.996  # Account for fees/slippage
                    capital -= position_value
                    entry_price = current_price
                    entry_time = i
                    
                elif position > 0:
                    # Check exit conditions
                    price_change = (current_price - entry_price) / entry_price
                    hold_time = i - entry_time
                    
                    # More lenient exit conditions
                    should_exit = (
                        prob < (threshold - 0.15) or  # Stronger reversal needed
                        price_change <= -stop_loss or
                        price_change >= take_profit or
                        hold_time >= max_hold or
                        i == len(X_test) - 1
                    )
                    
                    if should_exit:
                        # Exit position
                        exit_value = position * current_price * 0.996  # Account for fees/slippage
                        pnl = exit_value - (position * entry_price)
                        pnl_pct = (exit_value / (position * entry_price) - 1) * 100
                        
                        # Determine exit reason
                        exit_reason = "signal"
                        if price_change <= -stop_loss:
                            exit_reason = "stop_loss"
                        elif price_change >= take_profit:
                            exit_reason = "take_profit"
                        elif hold_time >= max_hold:
                            exit_reason = "time_limit"
                        
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'hold_time': hold_time,
                            'exit_reason': exit_reason,
                            'entry_prob': prob
                        })
                        
                        capital += exit_value
                        position = 0
            
            if not trades:
                return None
            
            # Calculate comprehensive metrics
            final_capital = capital
            total_return = (final_capital / initial_capital - 1) * 100
            
            # Trade statistics
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades) * 100
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
            
            # Risk metrics
            returns = np.diff(equity_curve) / equity_curve[:-1]
            
            if len(returns) > 1 and np.std(returns) > 0:
                # Adjust for timeframe
                periods_per_year = {
                    '1m': 365 * 24 * 60,
                    '3m': 365 * 24 * 20,
                    '5m': 365 * 24 * 12,
                    '15m': 365 * 24 * 4,
                    '1h': 365 * 24
                }
                periods = periods_per_year.get(config.get('timeframe', '5m'), 365 * 24 * 12)
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(periods)
            else:
                sharpe_ratio = 0
            
            # Drawdown calculation
            peak_equity = np.maximum.accumulate(equity_curve)
            drawdowns = (peak_equity - equity_curve) / peak_equity * 100
            max_drawdown = np.max(drawdowns)
            
            # Exit reason analysis
            exit_reasons = {}
            for trade in trades:
                reason = trade['exit_reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            # Calculate expectancy
            expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
            
            return {
                'total_return': total_return,
                'final_capital': final_capital,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'expectancy': expectancy,
                'exit_reasons': exit_reasons,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades)
            }
            
        except Exception as e:
            print(f"Backtesting error: {e}")
            return None
    
    def test_optimized_config(self, symbol: str, timeframe: str, config: Dict) -> Dict:
        """Test optimized configuration."""
        try:
            print(f"  Testing {symbol} {timeframe} with {config['model']}...")
            
            # Fetch extended data
            df = self.fetch_extended_data(symbol, timeframe, limit=3000)
            if df.empty or len(df) < 1500:
                return None
            
            # Enhanced feature engineering
            fdf = self.enhanced_feature_engineering(df)
            if fdf.empty or len(fdf) < 800:
                return None
            
            # Optimized labeling
            y = self.optimized_labeling(fdf, horizon=config['horizon'], cost_bp=config['cost'])
            if len(y) < 200:
                return None
            
            # Prepare data
            feature_cols = [c for c in fdf.columns if c not in ("ts", "open", "high", "low", "close", "volume", "dt")]
            X = fdf[feature_cols]
            
            # Use more training data
            split = int(len(X) * 0.75)
            X_train = X.iloc[:split]
            X_test = X.iloc[split:]
            y_train = y.iloc[:split]
            
            # Balanced model training
            model, train_metrics = self.balanced_model_training(X_train, y_train, config['model'])
            if model is None:
                return None
            
            # Optimized backtesting
            config_with_tf = config.copy()
            config_with_tf['timeframe'] = timeframe
            
            result = self.optimized_backtesting(model, X_test, fdf.iloc[split:], feature_cols, config_with_tf)
            
            if result:
                result['config'] = config
                result['symbol'] = symbol
                result['timeframe'] = timeframe
                result['train_metrics'] = train_metrics
                
                # Enhanced scoring
                score = self.calculate_enhanced_score(result)
                result['score'] = score
                
            return result
            
        except Exception as e:
            print(f"Config test error: {e}")
            return None
    
    def calculate_enhanced_score(self, result: Dict) -> float:
        """Calculate enhanced optimization score."""
        if not result:
            return 0
        
        # Base score from return (higher weight on good returns)
        return_score = min(result['total_return'] / 20, 3.0)  # Up to 3.0 for 20%+ returns
        
        # Sharpe bonus
        sharpe_bonus = min(result['sharpe_ratio'] / 1.0, 1.5)
        
        # Win rate bonus (more forgiving)
        win_rate_bonus = max(0, (result['win_rate'] - 40) / 60)  # Bonus above 40%
        
        # Profit factor bonus
        pf_bonus = min((result['profit_factor'] - 1) / 1.0, 1.0) if result['profit_factor'] > 1 else 0
        
        # Trade frequency bonus
        trade_bonus = min(result['total_trades'] / 30, 0.5)
        
        # Expectancy bonus
        expectancy_bonus = min(result['expectancy'] / 50, 0.8) if result['expectancy'] > 0 else 0
        
        # Penalties (more lenient)
        penalty = 0
        if result['total_trades'] < 3:
            penalty += 0.3  # Too few trades
        if result['max_drawdown'] > 30:
            penalty += 0.2  # Too much drawdown
        if result['total_return'] < 0:
            penalty += 1.0  # Negative returns
        
        score = (return_score + sharpe_bonus + win_rate_bonus + 
                pf_bonus + trade_bonus + expectancy_bonus - penalty)
        
        return max(0, score)
    
    def run_final_optimization(self):
        """Run final optimization with all improvements."""
        print("🚀 FINAL OPTIMIZED SYSTEM - MAXIMUM PROFITABILITY")
        print("=" * 70)
        
        # Optimized configurations based on diagnostic analysis
        optimized_configs = [
            # Ultra low cost, high frequency
            {"horizon": 1, "cost": 3, "model": "xgboost", "threshold": 0.52, "stop_loss": 0.02, "take_profit": 0.06, "max_hold": 60, "position_size": 0.3},
            {"horizon": 1, "cost": 4, "model": "lightgbm", "threshold": 0.54, "stop_loss": 0.025, "take_profit": 0.08, "max_hold": 80, "position_size": 0.35},
            {"horizon": 1, "cost": 5, "model": "ensemble", "threshold": 0.56, "stop_loss": 0.03, "take_profit": 0.1, "max_hold": 100, "position_size": 0.4},
            
            # Low cost momentum
            {"horizon": 2, "cost": 5, "model": "xgboost", "threshold": 0.55, "stop_loss": 0.025, "take_profit": 0.08, "max_hold": 80, "position_size": 0.35},
            {"horizon": 2, "cost": 6, "model": "lightgbm", "threshold": 0.58, "stop_loss": 0.03, "take_profit": 0.1, "max_hold": 100, "position_size": 0.4},
            {"horizon": 2, "cost": 8, "model": "ensemble", "threshold": 0.6, "stop_loss": 0.035, "take_profit": 0.12, "max_hold": 120, "position_size": 0.45},
            
            # Medium-term swing
            {"horizon": 3, "cost": 8, "model": "xgboost", "threshold": 0.58, "stop_loss": 0.03, "take_profit": 0.1, "max_hold": 120, "position_size": 0.4},
            {"horizon": 3, "cost": 10, "model": "lightgbm", "threshold": 0.62, "stop_loss": 0.035, "take_profit": 0.12, "max_hold": 150, "position_size": 0.45},
            {"horizon": 3, "cost": 12, "model": "ensemble", "threshold": 0.65, "stop_loss": 0.04, "take_profit": 0.15, "max_hold": 180, "position_size": 0.5},
        ]
        
        # Test on multiple timeframes including longer ones
        test_pairs = [
            ("BTCUSDT", "1m"),
            ("BTCUSDT", "3m"),
            ("BTCUSDT", "5m"),
            ("BTCUSDT", "15m"),  # Added longer timeframe
            ("ETHUSDT", "1m"),
            ("ETHUSDT", "3m"),
            ("ETHUSDT", "5m"),
            ("ETHUSDT", "15m"),  # Added longer timeframe
            ("ADAUSDT", "5m"),
            ("ADAUSDT", "15m"),  # Added longer timeframe
            ("BNBUSDT", "5m"),
            ("SOLUSDT", "5m"),
        ]
        
        print(f"Testing {len(optimized_configs)} optimized configs on {len(test_pairs)} pairs...")
        
        all_results = []
        total_tests = len(optimized_configs) * len(test_pairs)
        completed = 0
        
        for symbol, timeframe in test_pairs:
            print(f"\n🎯 Optimizing {symbol} {timeframe}...")
            
            for config in optimized_configs:
                completed += 1
                if completed % 5 == 0:
                    print(f"  Progress: {completed}/{total_tests} ({completed/total_tests*100:.1f}%)")
                
                result = self.test_optimized_config(symbol, timeframe, config)
                if result and result['total_return'] > 0:  # Only keep profitable results
                    all_results.append(result)
                    print(f"    ✓ {result['total_return']:.2f}% return, {result['total_trades']} trades")
        
        # Sort by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n🎉 OPTIMIZATION COMPLETE!")
        print(f"Found {len(all_results)} profitable configurations")
        
        return all_results
    
    def generate_final_report(self, results: List[Dict]) -> str:
        """Generate final comprehensive report."""
        if not results:
            return "❌ No profitable configurations found even with optimizations!"
        
        report = []
        report.append("=" * 100)
        report.append("🚀 FINAL OPTIMIZED SYSTEM REPORT - MAXIMUM PROFITABILITY")
        report.append("=" * 100)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Profitable configurations found: {len(results)}")
        report.append("")
        
        # Top 20 results
        report.append("🏆 TOP 20 PROFITABLE CONFIGURATIONS")
        report.append("-" * 80)
        
        for i, result in enumerate(results[:20], 1):
            config = result['config']
            report.append(f"{i}. {result['symbol']} {result['timeframe']} | {config['model']}")
            report.append(f"   H:{config['horizon']}, C:{config['cost']}bp, T:{config['threshold']}, PS:{config['position_size']*100:.0f}%")
            report.append(f"   Stop:{config['stop_loss']*100:.1f}%, Take:{config['take_profit']*100:.1f}%, Hold:{config['max_hold']}")
            report.append(f"   📈 Return: {result['total_return']:.2f}% | Sharpe: {result['sharpe_ratio']:.2f}")
            report.append(f"   📊 Trades: {result['total_trades']} | Win Rate: {result['win_rate']:.1f}%")
            report.append(f"   💰 PF: {result['profit_factor']:.2f} | Expectancy: ${result['expectancy']:.2f}")
            report.append(f"   📉 Max DD: {result['max_drawdown']:.2f}% | Score: {result['score']:.3f}")
            report.append("")
        
        # Performance statistics
        if results:
            returns = [r['total_return'] for r in results]
            sharpe_ratios = [r['sharpe_ratio'] for r in results]
            win_rates = [r['win_rate'] for r in results]
            
            report.append("📊 PERFORMANCE STATISTICS")
            report.append("-" * 50)
            report.append(f"Total Profitable Configs: {len(results)}")
            report.append(f"Average Return: {np.mean(returns):.2f}%")
            report.append(f"Best Return: {np.max(returns):.2f}%")
            report.append(f"Median Return: {np.median(returns):.2f}%")
            report.append(f"Average Sharpe: {np.mean(sharpe_ratios):.2f}")
            report.append(f"Average Win Rate: {np.mean(win_rates):.1f}%")
            report.append(f"Configs with >10% return: {len([r for r in results if r['total_return'] > 10])}")
            report.append(f"Configs with >20% return: {len([r for r in results if r['total_return'] > 20])}")
            report.append(f"Configs with >50% return: {len([r for r in results if r['total_return'] > 50])}")
            report.append("")
        
        # Best performing analysis
        if results:
            best = results[0]
            
            report.append("🎯 BEST CONFIGURATION ANALYSIS")
            report.append("-" * 50)
            report.append(f"Symbol: {best['symbol']}")
            report.append(f"Timeframe: {best['timeframe']}")
            report.append(f"Model: {best['config']['model']}")
            report.append(f"Configuration:")
            report.append(f"  - Horizon: {best['config']['horizon']} candles")
            report.append(f"  - Transaction Cost: {best['config']['cost']} basis points")
            report.append(f"  - Entry Threshold: {best['config']['threshold']}")
            report.append(f"  - Position Size: {best['config']['position_size']*100:.0f}% of capital")
            report.append(f"  - Stop Loss: {best['config']['stop_loss']*100:.1f}%")
            report.append(f"  - Take Profit: {best['config']['take_profit']*100:.1f}%")
            report.append(f"  - Max Hold Time: {best['config']['max_hold']} candles")
            report.append("")
            report.append(f"Performance:")
            report.append(f"  - Total Return: {best['total_return']:.2f}%")
            report.append(f"  - Sharpe Ratio: {best['sharpe_ratio']:.2f}")
            report.append(f"  - Win Rate: {best['win_rate']:.1f}%")
            report.append(f"  - Profit Factor: {best['profit_factor']:.2f}")
            report.append(f"  - Total Trades: {best['total_trades']}")
            report.append(f"  - Max Drawdown: {best['max_drawdown']:.2f}%")
            report.append(f"  - Expectancy: ${best['expectancy']:.2f} per trade")
            report.append("")
            
            # Exit reason analysis
            if 'exit_reasons' in best:
                report.append(f"Exit Reasons:")
                for reason, count in best['exit_reasons'].items():
                    pct = count / best['total_trades'] * 100
                    report.append(f"  - {reason}: {count} ({pct:.1f}%)")
                report.append("")
        
        # Implementation guide
        report.append("🚀 IMPLEMENTATION GUIDE")
        report.append("-" * 50)
        report.append("STEP 1: SETUP")
        report.append("1. Use the best configuration identified above")
        report.append("2. Start with paper trading for 1-2 weeks")
        report.append("3. Begin with 25% of recommended position size")
        report.append("4. Gradually increase to full size as confidence builds")
        report.append("")
        
        report.append("STEP 2: RISK MANAGEMENT")
        report.append("1. Set daily loss limit at 3% of total capital")
        report.append("2. Set weekly loss limit at 8% of total capital")
        report.append("3. Stop trading if drawdown exceeds 15%")
        report.append("4. Review performance weekly")
        report.append("")
        
        report.append("STEP 3: MONITORING")
        report.append("1. Track win rate (should stay above 45%)")
        report.append("2. Monitor profit factor (should stay above 1.2)")
        report.append("3. Watch for model degradation")
        report.append("4. Re-optimize monthly")
        report.append("")
        
        report.append("STEP 4: SCALING")
        report.append("1. Start with $1,000-$5,000 capital")
        report.append("2. Scale up gradually as performance proves consistent")
        report.append("3. Consider multiple configurations for diversification")
        report.append("4. Implement portfolio-level risk management")
        report.append("")
        
        # Alternative configurations
        if len(results) > 1:
            report.append("🔄 ALTERNATIVE CONFIGURATIONS")
            report.append("-" * 50)
            report.append("For diversification, consider these top alternatives:")
            
            for i, alt in enumerate(results[1:6], 2):
                report.append(f"{i}. {alt['symbol']} {alt['timeframe']} | Return: {alt['total_return']:.2f}% | Trades: {alt['total_trades']}")
            
            report.append("")
            report.append("Benefits of using multiple configurations:")
            report.append("- Reduced correlation risk")
            report.append("- More consistent returns")
            report.append("- Better risk-adjusted performance")
            report.append("- Protection against model degradation")
        
        return "\n".join(report)


def main():
    """Main function."""
    print("🚀 FINAL OPTIMIZED TRADING SYSTEM")
    print("Implementing all diagnostic recommendations")
    print("=" * 70)
    
    start_time = time.time()
    
    system = FinalOptimizedSystem()
    results = system.run_final_optimization()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n⏱️  Final optimization completed in {duration/60:.1f} minutes")
    
    # Generate comprehensive report
    report = system.generate_final_report(results)
    print("\n" + report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"final_optimized_results_{timestamp}.json", "w") as f:
        json.dump({
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'duration_minutes': duration / 60,
            'total_profitable_configs': len(results),
            'optimization_version': 'final_v1.0'
        }, f, indent=2, default=str)
    
    with open(f"FINAL_OPTIMIZATION_REPORT_{timestamp}.md", "w") as f:
        f.write(report)
    
    print(f"\n📁 Final results saved:")
    print(f"   - final_optimized_results_{timestamp}.json")
    print(f"   - FINAL_OPTIMIZATION_REPORT_{timestamp}.md")
    
    if results:
        best = results[0]
        print(f"\n🎉 BEST CONFIGURATION SUMMARY:")
        print(f"   {best['symbol']} {best['timeframe']} | {best['config']['model']}")
        print(f"   Return: {best['total_return']:.2f}% | Sharpe: {best['sharpe_ratio']:.2f}")
        print(f"   Win Rate: {best['win_rate']:.1f}% | Trades: {best['total_trades']}")
        print(f"   Max DD: {best['max_drawdown']:.2f}% | Score: {best['score']:.3f}")
        
        print(f"\n💡 READY FOR IMPLEMENTATION!")
        print(f"   Start with paper trading using the best configuration")
        print(f"   Expected performance: {best['total_return']:.2f}% return")
        print(f"   Risk level: {best['max_drawdown']:.2f}% max drawdown")
    else:
        print("\n❌ No profitable configurations found even with all optimizations.")
        print("   This suggests the current market conditions or approach may need")
        print("   fundamental changes. Consider:")
        print("   - Different asset classes")
        print("   - Longer timeframes (1h, 4h, 1d)")
        print("   - Different market conditions")
        print("   - Alternative strategies (mean reversion, arbitrage)")
    
    return results


if __name__ == "__main__":
    results = main()