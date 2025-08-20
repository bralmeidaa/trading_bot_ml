#!/usr/bin/env python3
"""
Diagnostic analysis to understand why configurations are not profitable.
Deep dive into the data and model performance.
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from data.binance_client import BinancePublicClient
from ml.features import build_features, build_labels
from ml.models import create_model


class DiagnosticAnalyzer:
    """Diagnostic analysis for trading system."""
    
    def __init__(self):
        self.client = BinancePublicClient()
        
    def analyze_data_quality(self, symbol: str, timeframe: str) -> Dict:
        """Analyze data quality and characteristics."""
        print(f"\n🔍 Analyzing {symbol} {timeframe} data quality...")
        
        try:
            # Fetch data
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=2000)
            if not ohlcv:
                return {"error": "No data available"}
            
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            
            # Basic statistics
            analysis = {
                'total_candles': len(df),
                'date_range': {
                    'start': pd.to_datetime(df['ts'].min(), unit='ms').strftime('%Y-%m-%d %H:%M'),
                    'end': pd.to_datetime(df['ts'].max(), unit='ms').strftime('%Y-%m-%d %H:%M')
                },
                'price_stats': {
                    'min': df['close'].min(),
                    'max': df['close'].max(),
                    'mean': df['close'].mean(),
                    'std': df['close'].std(),
                    'cv': df['close'].std() / df['close'].mean()  # Coefficient of variation
                },
                'volume_stats': {
                    'min': df['volume'].min(),
                    'max': df['volume'].max(),
                    'mean': df['volume'].mean(),
                    'zero_volume_candles': (df['volume'] == 0).sum()
                }
            }
            
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            df['abs_returns'] = df['returns'].abs()
            
            analysis['return_stats'] = {
                'mean_return': df['returns'].mean(),
                'std_return': df['returns'].std(),
                'skewness': df['returns'].skew(),
                'kurtosis': df['returns'].kurtosis(),
                'positive_returns': (df['returns'] > 0).sum(),
                'negative_returns': (df['returns'] < 0).sum(),
                'zero_returns': (df['returns'] == 0).sum()
            }
            
            # Volatility analysis
            df['volatility'] = df['abs_returns'].rolling(20).mean()
            analysis['volatility_stats'] = {
                'mean_volatility': df['volatility'].mean(),
                'volatility_std': df['volatility'].std(),
                'high_vol_periods': (df['volatility'] > df['volatility'].quantile(0.9)).sum()
            }
            
            # Trend analysis
            df['sma_20'] = df['close'].rolling(20).mean()
            df['trend'] = np.where(df['close'] > df['sma_20'], 1, -1)
            
            trend_changes = (df['trend'].diff() != 0).sum()
            analysis['trend_stats'] = {
                'trend_changes': trend_changes,
                'uptrend_periods': (df['trend'] == 1).sum(),
                'downtrend_periods': (df['trend'] == -1).sum(),
                'trend_persistence': len(df) / max(trend_changes, 1)
            }
            
            # Gap analysis
            df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            analysis['gap_stats'] = {
                'mean_gap': df['gap'].mean(),
                'std_gap': df['gap'].std(),
                'significant_gaps': (df['gap'].abs() > 0.01).sum()  # >1% gaps
            }
            
            print(f"✓ Data Quality Analysis Complete")
            print(f"  Candles: {analysis['total_candles']}")
            print(f"  Mean Return: {analysis['return_stats']['mean_return']*100:.4f}%")
            print(f"  Volatility: {analysis['return_stats']['std_return']*100:.2f}%")
            print(f"  Trend Changes: {analysis['trend_stats']['trend_changes']}")
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_feature_quality(self, symbol: str, timeframe: str) -> Dict:
        """Analyze feature engineering quality."""
        print(f"\n🔧 Analyzing feature quality for {symbol} {timeframe}...")
        
        try:
            # Fetch data
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=1500)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            
            # Build features
            fdf = build_features(df, advanced=True)
            if fdf.empty:
                return {"error": "No features generated"}
            
            # Analyze features
            feature_cols = [c for c in fdf.columns if c not in ("ts", "open", "high", "low", "close", "volume", "dt")]
            
            analysis = {
                'total_features': len(feature_cols),
                'total_samples': len(fdf),
                'feature_stats': {}
            }
            
            # Analyze each feature
            for col in feature_cols:
                if col in fdf.columns:
                    series = fdf[col]
                    analysis['feature_stats'][col] = {
                        'mean': series.mean() if not series.isna().all() else None,
                        'std': series.std() if not series.isna().all() else None,
                        'null_count': series.isna().sum(),
                        'null_percentage': series.isna().sum() / len(series) * 100,
                        'unique_values': series.nunique(),
                        'zero_values': (series == 0).sum() if not series.isna().all() else 0
                    }
            
            # Feature quality metrics
            high_null_features = [f for f, stats in analysis['feature_stats'].items() 
                                if stats['null_percentage'] > 50]
            
            low_variance_features = [f for f, stats in analysis['feature_stats'].items() 
                                   if stats['std'] is not None and stats['std'] < 0.001]
            
            analysis['quality_issues'] = {
                'high_null_features': high_null_features,
                'low_variance_features': low_variance_features,
                'features_with_issues': len(high_null_features) + len(low_variance_features)
            }
            
            print(f"✓ Feature Analysis Complete")
            print(f"  Total Features: {analysis['total_features']}")
            print(f"  Samples: {analysis['total_samples']}")
            print(f"  High Null Features: {len(high_null_features)}")
            print(f"  Low Variance Features: {len(low_variance_features)}")
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_label_distribution(self, symbol: str, timeframe: str) -> Dict:
        """Analyze label distribution and quality."""
        print(f"\n🏷️  Analyzing label distribution for {symbol} {timeframe}...")
        
        try:
            # Fetch data
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=1500)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            
            # Build features
            fdf = build_features(df, advanced=False)  # Use basic features for speed
            if fdf.empty:
                return {"error": "No features generated"}
            
            analysis = {}
            
            # Test different label configurations
            label_configs = [
                {"horizon": 1, "cost": 5},
                {"horizon": 1, "cost": 10},
                {"horizon": 1, "cost": 15},
                {"horizon": 2, "cost": 10},
                {"horizon": 3, "cost": 15},
                {"horizon": 5, "cost": 20}
            ]
            
            for config in label_configs:
                try:
                    y = build_labels(fdf, horizon=config['horizon'], cost_bp=config['cost'])
                    if len(y) > 0:
                        config_key = f"h{config['horizon']}_c{config['cost']}"
                        
                        # Convert to int for analysis
                        y_int = y.astype(int)
                        
                        analysis[config_key] = {
                            'total_labels': len(y),
                            'positive_labels': (y_int == 1).sum(),
                            'negative_labels': (y_int == 0).sum(),
                            'positive_percentage': (y_int == 1).sum() / len(y) * 100,
                            'class_balance': min((y_int == 1).sum(), (y_int == 0).sum()) / max((y_int == 1).sum(), (y_int == 0).sum())
                        }
                except Exception as e:
                    analysis[f"h{config['horizon']}_c{config['cost']}"] = {"error": str(e)}
            
            print(f"✓ Label Analysis Complete")
            for config_key, stats in analysis.items():
                if 'error' not in stats:
                    print(f"  {config_key}: {stats['positive_percentage']:.1f}% positive, balance: {stats['class_balance']:.3f}")
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_model_performance(self, symbol: str, timeframe: str) -> Dict:
        """Analyze model performance in detail."""
        print(f"\n🤖 Analyzing model performance for {symbol} {timeframe}...")
        
        try:
            # Fetch data
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=1500)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            
            # Build features
            fdf = build_features(df, advanced=False)
            if fdf.empty:
                return {"error": "No features generated"}
            
            # Build labels
            y = build_labels(fdf, horizon=2, cost_bp=10)
            if len(y) == 0:
                return {"error": "No labels generated"}
            
            # Prepare data
            feature_cols = [c for c in fdf.columns if c not in ("ts", "open", "high", "low", "close", "volume", "dt")]
            X = fdf[feature_cols]
            
            if len(X) < 200:
                return {"error": "Insufficient data"}
            
            # Split data
            split = int(len(X) * 0.7)
            X_train = X.iloc[:split]
            X_test = X.iloc[split:]
            y_train = y.iloc[:split]
            y_test = y.iloc[split:]
            
            analysis = {}
            
            # Test different models
            models = ["xgboost", "lightgbm", "ensemble"]
            
            for model_type in models:
                try:
                    print(f"  Testing {model_type}...")
                    
                    # Train model
                    model = create_model(model_type, max_features=20)
                    train_metrics = model.fit(X_train, y_train.values)
                    
                    # Predictions
                    y_pred_proba = model.predict_proba(X_test)
                    y_pred = model.predict(X_test, threshold=0.6)
                    
                    # Calculate metrics
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                    
                    accuracy = accuracy_score(y_test.values, y_pred)
                    precision = precision_score(y_test.values, y_pred, zero_division=0)
                    recall = recall_score(y_test.values, y_pred, zero_division=0)
                    f1 = f1_score(y_test.values, y_pred, zero_division=0)
                    
                    try:
                        auc = roc_auc_score(y_test.values, y_pred_proba)
                    except:
                        auc = 0.5
                    
                    # Prediction distribution
                    pred_dist = {
                        'total_predictions': len(y_pred),
                        'positive_predictions': (y_pred == 1).sum(),
                        'negative_predictions': (y_pred == 0).sum(),
                        'positive_percentage': (y_pred == 1).sum() / len(y_pred) * 100
                    }
                    
                    # Probability distribution
                    prob_stats = {
                        'mean_prob': np.mean(y_pred_proba),
                        'std_prob': np.std(y_pred_proba),
                        'min_prob': np.min(y_pred_proba),
                        'max_prob': np.max(y_pred_proba),
                        'high_confidence_predictions': (y_pred_proba > 0.7).sum()
                    }
                    
                    analysis[model_type] = {
                        'train_metrics': train_metrics,
                        'test_metrics': {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'auc': auc
                        },
                        'prediction_distribution': pred_dist,
                        'probability_stats': prob_stats
                    }
                    
                except Exception as e:
                    analysis[model_type] = {"error": str(e)}
            
            print(f"✓ Model Analysis Complete")
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_trading_simulation(self, symbol: str, timeframe: str) -> Dict:
        """Simulate trading with detailed analysis."""
        print(f"\n💹 Analyzing trading simulation for {symbol} {timeframe}...")
        
        try:
            # Fetch data
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=1500)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            
            # Build features
            fdf = build_features(df, advanced=False)
            if fdf.empty:
                return {"error": "No features generated"}
            
            # Build labels
            y = build_labels(fdf, horizon=2, cost_bp=10)
            if len(y) == 0:
                return {"error": "No labels generated"}
            
            # Prepare data
            feature_cols = [c for c in fdf.columns if c not in ("ts", "open", "high", "low", "close", "volume", "dt")]
            X = fdf[feature_cols]
            
            # Split data
            split = int(len(X) * 0.7)
            X_train = X.iloc[:split]
            X_test = X.iloc[split:]
            y_train = y.iloc[:split]
            
            # Train model
            model = create_model("xgboost", max_features=20)
            model.fit(X_train, y_train.values)
            
            # Simulate trading
            price_data = fdf.iloc[split:]
            
            # Different threshold tests
            thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
            simulation_results = {}
            
            for threshold in thresholds:
                capital = 10000.0
                position = 0
                trades = []
                signals = []
                
                for i, (_, row) in enumerate(X_test.iterrows()):
                    current_price = price_data.iloc[i]['close']
                    
                    # Get prediction
                    features = row[feature_cols].values.reshape(1, -1)
                    features_df = pd.DataFrame(features, columns=feature_cols)
                    prob = model.predict_proba(features_df)[0]
                    
                    signals.append({
                        'index': i,
                        'price': current_price,
                        'probability': prob,
                        'signal': 1 if prob > threshold else 0
                    })
                    
                    if position == 0 and prob > threshold:
                        # Enter position
                        position = (capital * 0.1) / current_price * 0.998
                        entry_price = current_price
                        entry_idx = i
                        
                    elif position > 0:
                        # Check exit (simple exit after 5 candles or signal reversal)
                        if prob < (threshold - 0.1) or (i - entry_idx) >= 5 or i == len(X_test) - 1:
                            # Exit position
                            pnl = position * (current_price - entry_price) * 0.998
                            trades.append({
                                'entry_price': entry_price,
                                'exit_price': current_price,
                                'pnl': pnl,
                                'pnl_pct': (current_price / entry_price - 1) * 100,
                                'hold_time': i - entry_idx,
                                'entry_prob': prob
                            })
                            position = 0
                
                # Calculate results
                total_pnl = sum(t['pnl'] for t in trades)
                total_return = (total_pnl / 10000) * 100
                
                signal_stats = {
                    'total_signals': len([s for s in signals if s['signal'] == 1]),
                    'signal_rate': len([s for s in signals if s['signal'] == 1]) / len(signals) * 100,
                    'avg_probability': np.mean([s['probability'] for s in signals]),
                    'high_prob_signals': len([s for s in signals if s['probability'] > 0.7])
                }
                
                simulation_results[f"threshold_{threshold}"] = {
                    'total_return': total_return,
                    'total_trades': len(trades),
                    'total_pnl': total_pnl,
                    'signal_stats': signal_stats,
                    'trades': trades[:5]  # First 5 trades for analysis
                }
            
            print(f"✓ Trading Simulation Complete")
            for thresh, results in simulation_results.items():
                print(f"  {thresh}: {results['total_return']:.2f}% return, {results['total_trades']} trades")
            
            return simulation_results
            
        except Exception as e:
            return {"error": str(e)}
    
    def run_comprehensive_diagnostic(self):
        """Run comprehensive diagnostic analysis."""
        print("🔍 COMPREHENSIVE DIAGNOSTIC ANALYSIS")
        print("=" * 60)
        
        # Test symbols and timeframes
        test_cases = [
            ("BTCUSDT", "1m"),
            ("BTCUSDT", "5m"),
            ("ETHUSDT", "5m")
        ]
        
        full_analysis = {}
        
        for symbol, timeframe in test_cases:
            print(f"\n{'='*60}")
            print(f"ANALYZING {symbol} {timeframe}")
            print(f"{'='*60}")
            
            case_analysis = {}
            
            # Data quality analysis
            case_analysis['data_quality'] = self.analyze_data_quality(symbol, timeframe)
            
            # Feature quality analysis
            case_analysis['feature_quality'] = self.analyze_feature_quality(symbol, timeframe)
            
            # Label distribution analysis
            case_analysis['label_distribution'] = self.analyze_label_distribution(symbol, timeframe)
            
            # Model performance analysis
            case_analysis['model_performance'] = self.analyze_model_performance(symbol, timeframe)
            
            # Trading simulation
            case_analysis['trading_simulation'] = self.analyze_trading_simulation(symbol, timeframe)
            
            full_analysis[f"{symbol}_{timeframe}"] = case_analysis
        
        return full_analysis
    
    def generate_diagnostic_report(self, analysis: Dict) -> str:
        """Generate comprehensive diagnostic report."""
        report = []
        report.append("=" * 80)
        report.append("🔍 COMPREHENSIVE DIAGNOSTIC ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for case_key, case_analysis in analysis.items():
            symbol, timeframe = case_key.split('_')
            
            report.append(f"📊 ANALYSIS: {symbol} {timeframe}")
            report.append("-" * 50)
            
            # Data Quality Summary
            if 'data_quality' in case_analysis and 'error' not in case_analysis['data_quality']:
                dq = case_analysis['data_quality']
                report.append(f"Data Quality:")
                report.append(f"  Candles: {dq['total_candles']}")
                report.append(f"  Mean Return: {dq['return_stats']['mean_return']*100:.4f}%")
                report.append(f"  Volatility: {dq['return_stats']['std_return']*100:.2f}%")
                report.append(f"  Trend Changes: {dq['trend_stats']['trend_changes']}")
                report.append("")
            
            # Feature Quality Summary
            if 'feature_quality' in case_analysis and 'error' not in case_analysis['feature_quality']:
                fq = case_analysis['feature_quality']
                report.append(f"Feature Quality:")
                report.append(f"  Total Features: {fq['total_features']}")
                report.append(f"  Samples: {fq['total_samples']}")
                report.append(f"  High Null Features: {len(fq['quality_issues']['high_null_features'])}")
                report.append(f"  Low Variance Features: {len(fq['quality_issues']['low_variance_features'])}")
                report.append("")
            
            # Label Distribution Summary
            if 'label_distribution' in case_analysis:
                ld = case_analysis['label_distribution']
                report.append(f"Label Distribution:")
                for config, stats in ld.items():
                    if 'error' not in stats:
                        report.append(f"  {config}: {stats['positive_percentage']:.1f}% positive")
                report.append("")
            
            # Model Performance Summary
            if 'model_performance' in case_analysis:
                mp = case_analysis['model_performance']
                report.append(f"Model Performance:")
                for model, stats in mp.items():
                    if 'error' not in stats:
                        test_metrics = stats['test_metrics']
                        report.append(f"  {model}: Acc {test_metrics['accuracy']:.3f}, AUC {test_metrics['auc']:.3f}")
                report.append("")
            
            # Trading Simulation Summary
            if 'trading_simulation' in case_analysis and 'error' not in case_analysis['trading_simulation']:
                ts = case_analysis['trading_simulation']
                report.append(f"Trading Simulation:")
                best_threshold = max(ts.items(), key=lambda x: x[1]['total_return'])
                report.append(f"  Best Threshold: {best_threshold[0]} ({best_threshold[1]['total_return']:.2f}% return)")
                report.append(f"  Best Trades: {best_threshold[1]['total_trades']}")
                report.append("")
            
            report.append("")
        
        # Overall conclusions
        report.append("💡 KEY FINDINGS & RECOMMENDATIONS")
        report.append("-" * 50)
        
        # Analyze common issues
        common_issues = []
        
        # Check for low returns across all cases
        all_returns = []
        for case_analysis in analysis.values():
            if 'trading_simulation' in case_analysis and 'error' not in case_analysis['trading_simulation']:
                ts = case_analysis['trading_simulation']
                best_return = max([results['total_return'] for results in ts.values()])
                all_returns.append(best_return)
        
        if all_returns and max(all_returns) < 5:
            common_issues.append("LOW PROFITABILITY: All configurations show <5% returns")
        
        # Check for feature quality issues
        feature_issues = 0
        for case_analysis in analysis.values():
            if 'feature_quality' in case_analysis and 'error' not in case_analysis['feature_quality']:
                fq = case_analysis['feature_quality']
                if fq['quality_issues']['features_with_issues'] > fq['total_features'] * 0.3:
                    feature_issues += 1
        
        if feature_issues > len(analysis) * 0.5:
            common_issues.append("FEATURE QUALITY: >30% of features have quality issues")
        
        # Check for label imbalance
        label_issues = 0
        for case_analysis in analysis.values():
            if 'label_distribution' in case_analysis:
                ld = case_analysis['label_distribution']
                for stats in ld.values():
                    if 'error' not in stats and stats['class_balance'] < 0.3:
                        label_issues += 1
                        break
        
        if label_issues > 0:
            common_issues.append("LABEL IMBALANCE: Severe class imbalance detected")
        
        # Report issues and recommendations
        if common_issues:
            report.append("🚨 IDENTIFIED ISSUES:")
            for issue in common_issues:
                report.append(f"  - {issue}")
            report.append("")
        
        report.append("🔧 RECOMMENDATIONS:")
        
        if "LOW PROFITABILITY" in str(common_issues):
            report.append("  1. REDUCE TRANSACTION COSTS: Try lower cost_bp values (3-8)")
            report.append("  2. INCREASE POSITION SIZE: Test 30-50% position sizes")
            report.append("  3. OPTIMIZE THRESHOLDS: Try lower entry thresholds (0.52-0.58)")
            report.append("  4. EXTEND HOLDING PERIODS: Allow longer trades (50-100 candles)")
        
        if "FEATURE QUALITY" in str(common_issues):
            report.append("  5. FEATURE SELECTION: Remove high-null and low-variance features")
            report.append("  6. FEATURE ENGINEERING: Add momentum and volatility features")
            report.append("  7. DATA PREPROCESSING: Implement better normalization")
        
        if "LABEL IMBALANCE" in str(common_issues):
            report.append("  8. REBALANCING: Use SMOTE or class weights")
            report.append("  9. THRESHOLD TUNING: Adjust for imbalanced classes")
            report.append("  10. DIFFERENT LABELING: Try regression instead of classification")
        
        report.append("")
        report.append("🎯 NEXT STEPS:")
        report.append("  1. Implement recommended changes")
        report.append("  2. Test with longer timeframes (15m, 1h)")
        report.append("  3. Try different symbols with higher volatility")
        report.append("  4. Consider ensemble of multiple timeframes")
        report.append("  5. Implement dynamic position sizing")
        
        return "\n".join(report)


def main():
    """Main diagnostic function."""
    print("🔍 COMPREHENSIVE DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    
    analyzer = DiagnosticAnalyzer()
    analysis = analyzer.run_comprehensive_diagnostic()
    
    # Generate report
    report = analyzer.generate_diagnostic_report(analysis)
    print("\n" + report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"diagnostic_analysis_{timestamp}.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    
    with open(f"DIAGNOSTIC_REPORT_{timestamp}.md", "w") as f:
        f.write(report)
    
    print(f"\n📁 Results saved:")
    print(f"   - diagnostic_analysis_{timestamp}.json")
    print(f"   - DIAGNOSTIC_REPORT_{timestamp}.md")
    
    return analysis


if __name__ == "__main__":
    analysis = main()