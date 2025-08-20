#!/usr/bin/env python3
"""
Comprehensive profitability validation script for the trading bot ML system.
Tests multiple strategies, models, and market conditions to validate effectiveness.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
try:
    from backend.data.binance_client import BinancePublicClient
    from backend.ml.features import build_features, build_labels
    from backend.ml.models import create_model
    from backend.ml.advanced_models import create_default_ensemble, optimize_model_hyperparameters
    from backend.backtesting.advanced_engine import AdvancedBacktester, BacktestConfig, BacktestResults
    from backend.core.advanced_risk import AdvancedRiskParams, create_conservative_risk_params, create_aggressive_risk_params
    from backend.optimization.walk_forward import WalkForwardOptimizer, WalkForwardConfig, MonteCarloAnalyzer
    from backend.strategy.ml_prob import MLProbStrategy
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import method...")
    
    # Alternative import method
    import importlib.util
    
    def import_module_from_path(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class ProfitabilityValidator:
    """Comprehensive profitability validation system."""
    
    def __init__(self):
        self.client = BinancePublicClient()
        self.results = {}
    
    def fetch_test_data(self, symbols: list = None, timeframe: str = "1h", days: int = 365) -> dict:
        """Fetch test data for multiple symbols."""
        if symbols is None:
            symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        
        print(f"Fetching {days} days of {timeframe} data for {len(symbols)} symbols...")
        
        data = {}
        for symbol in symbols:
            try:
                df = self.client.get_klines(symbol, timeframe, limit=days * 24)
                if not df.empty:
                    data[symbol] = df
                    print(f"✓ {symbol}: {len(df)} candles")
                else:
                    print(f"✗ {symbol}: No data")
            except Exception as e:
                print(f"✗ {symbol}: Error - {e}")
        
        return data
    
    def test_basic_ml_strategy(self, data: dict) -> dict:
        """Test basic ML strategy performance."""
        print("\n=== Testing Basic ML Strategy ===")
        
        results = {}
        
        for symbol, df in data.items():
            print(f"\nTesting {symbol}...")
            
            try:
                # Test different model types
                model_types = ["ensemble", "xgboost", "lightgbm", "random_forest"]
                
                for model_type in model_types:
                    print(f"  Testing {model_type}...")
                    
                    strategy = MLProbStrategy(
                        model_type=model_type,
                        use_advanced_features=True,
                        feature_selection=True,
                        max_features=30,
                        threshold=0.6,
                        train_ratio=0.7
                    )
                    
                    result = strategy.run(df)
                    
                    if result.trades:
                        total_return = ((result.equity_curve[-1]["equity"] / 1000.0) - 1) * 100
                        win_rate = len([t for t in result.trades if t["pnl"] > 0]) / len(result.trades) * 100
                        
                        results[f"{symbol}_{model_type}"] = {
                            "total_return": total_return,
                            "total_trades": len(result.trades),
                            "win_rate": win_rate,
                            "sharpe": result.metrics.get("sharpe", 0),
                            "max_drawdown": result.metrics.get("max_drawdown", 0)
                        }
                        
                        print(f"    Return: {total_return:.2f}%, Trades: {len(result.trades)}, Win Rate: {win_rate:.1f}%")
                    else:
                        print(f"    No trades generated")
                        
            except Exception as e:
                print(f"  Error testing {symbol}: {e}")
        
        return results
    
    def test_advanced_backtesting(self, data: dict) -> dict:
        """Test advanced backtesting engine."""
        print("\n=== Testing Advanced Backtesting Engine ===")
        
        results = {}
        
        # Test with different risk parameters
        risk_configs = {
            "conservative": create_conservative_risk_params(),
            "moderate": AdvancedRiskParams(),
            "aggressive": create_aggressive_risk_params()
        }
        
        for symbol, df in list(data.items())[:2]:  # Test on first 2 symbols
            print(f"\nTesting {symbol}...")
            
            # Prepare features
            fdf = build_features(df, advanced=True)
            if fdf.empty:
                continue
            
            # Build labels
            y = build_labels(fdf, horizon=5, cost_bp=15.0)
            
            # Split data
            split = int(len(fdf) * 0.7)
            X_train = fdf.iloc[:split]
            y_train = y.iloc[:split]
            X_test = fdf.iloc[split:]
            
            # Train model
            feature_cols = [c for c in X_train.columns if c not in ("ts", "open", "high", "low", "close", "volume", "dt")]
            model = create_model("ensemble")
            
            try:
                train_metrics = model.fit(X_train[feature_cols], y_train.values)
                print(f"  Model trained: {train_metrics.get('n_features', 0)} features")
                
                for risk_name, risk_params in risk_configs.items():
                    print(f"  Testing {risk_name} risk management...")
                    
                    # Configure backtester
                    config = BacktestConfig(
                        initial_capital=10000.0,
                        commission_pct=0.1,
                        slippage_pct=0.05,
                        use_advanced_risk=True,
                        risk_params=risk_params
                    )
                    
                    backtester = AdvancedBacktester(config)
                    backtester.reset()
                    
                    # Run backtest
                    for _, row in X_test.iterrows():
                        timestamp = int(row['ts'])
                        price = float(row['close'])
                        
                        # Get prediction
                        features = row[feature_cols].values.reshape(1, -1)
                        prob = model.predict_proba(pd.DataFrame(features, columns=feature_cols))[0]
                        
                        # Generate signals
                        if prob > 0.6:
                            backtester.place_order('buy', 100, price, timestamp, confidence=prob)
                        elif len(backtester.positions) > 0 and prob < 0.4:
                            backtester.place_order('sell', 100, price, timestamp)
                        
                        # Update backtester
                        backtester.update(timestamp, price, row.get('volume', 0))
                    
                    # Finalize and get results
                    if not X_test.empty:
                        final_row = X_test.iloc[-1]
                        backtester.finalize(float(final_row['close']), int(final_row['ts']))
                    
                    start_date = pd.to_datetime(X_test['ts'].min(), unit='ms')
                    end_date = pd.to_datetime(X_test['ts'].max(), unit='ms')
                    backtest_results = backtester.get_results(start_date, end_date)
                    
                    results[f"{symbol}_{risk_name}"] = {
                        "total_return": backtest_results.total_return,
                        "annual_return": backtest_results.annual_return,
                        "sharpe_ratio": backtest_results.sharpe_ratio,
                        "max_drawdown": backtest_results.max_drawdown,
                        "total_trades": backtest_results.total_trades,
                        "win_rate": backtest_results.win_rate,
                        "profit_factor": backtest_results.profit_factor
                    }
                    
                    print(f"    Return: {backtest_results.total_return:.2f}%, Sharpe: {backtest_results.sharpe_ratio:.2f}")
                    
            except Exception as e:
                print(f"  Error: {e}")
        
        return results
    
    def test_walk_forward_optimization(self, data: dict) -> dict:
        """Test walk-forward optimization."""
        print("\n=== Testing Walk-Forward Optimization ===")
        
        results = {}
        
        # Test on one symbol (most computationally intensive)
        symbol = list(data.keys())[0]
        df = data[symbol]
        
        print(f"Testing walk-forward optimization on {symbol}...")
        
        try:
            # Configure walk-forward
            wf_config = WalkForwardConfig(
                training_window_days=90,  # 3 months training
                testing_window_days=15,   # 2 weeks testing
                step_size_days=15,        # Move forward by 2 weeks
                optimization_metric="sharpe_ratio",
                max_iterations=20,        # Limit for speed
                monte_carlo_runs=100      # Reduced for speed
            )
            
            # Configure backtesting
            backtest_config = BacktestConfig(
                initial_capital=10000.0,
                commission_pct=0.1,
                use_advanced_risk=True,
                risk_params=AdvancedRiskParams()
            )
            
            # Parameter ranges to optimize
            parameter_ranges = {
                'threshold': [0.55, 0.6, 0.65, 0.7],
                'model_type': ['xgboost', 'lightgbm'],
                'max_features': [20, 30, 40]
            }
            
            # Create a simple strategy class for testing
            class SimpleMLStrategy:
                def __init__(self, threshold=0.6, model_type='xgboost', max_features=30):
                    self.threshold = threshold
                    self.model_type = model_type
                    self.max_features = max_features
                    self.model = None
                    self.feature_cols = []
                
                def get_signal(self, row):
                    # Simplified signal generation for testing
                    if self.model is None:
                        return 'hold'
                    
                    try:
                        features = row[self.feature_cols].values.reshape(1, -1)
                        prob = self.model.predict_proba(pd.DataFrame(features, columns=self.feature_cols))[0]
                        
                        if prob > self.threshold:
                            return 'buy'
                        elif prob < (1 - self.threshold):
                            return 'sell'
                        else:
                            return 'hold'
                    except:
                        return 'hold'
            
            # Run walk-forward optimization
            optimizer = WalkForwardOptimizer(wf_config)
            
            # This is a simplified version - in practice you'd need to implement
            # the full integration with your strategy classes
            print("  Walk-forward optimization would run here...")
            print("  (Simplified for demonstration)")
            
            results[symbol] = {
                "status": "simulated",
                "message": "Walk-forward optimization framework ready"
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            results[symbol] = {"error": str(e)}
        
        return results
    
    def test_feature_importance(self, data: dict) -> dict:
        """Test feature importance and selection."""
        print("\n=== Testing Feature Importance ===")
        
        results = {}
        
        for symbol, df in list(data.items())[:2]:  # Test first 2 symbols
            print(f"\nAnalyzing features for {symbol}...")
            
            try:
                # Build features
                fdf = build_features(df, advanced=True)
                if fdf.empty:
                    continue
                
                # Build labels
                y = build_labels(fdf, horizon=5, cost_bp=15.0)
                
                # Get feature columns
                feature_cols = [c for c in fdf.columns if c not in ("ts", "open", "high", "low", "close", "volume", "dt")]
                X = fdf[feature_cols]
                
                print(f"  Total features: {len(feature_cols)}")
                
                # Test different models for feature importance
                model_types = ["xgboost", "lightgbm", "random_forest"]
                
                for model_type in model_types:
                    print(f"  Testing {model_type} feature importance...")
                    
                    model = create_model(model_type, feature_selection=False)
                    train_metrics = model.fit(X, y.values)
                    
                    # Get feature importance
                    importance = model.get_feature_importance()
                    
                    if importance:
                        # Top 10 features
                        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                        
                        results[f"{symbol}_{model_type}"] = {
                            "total_features": len(feature_cols),
                            "top_features": top_features,
                            "train_auc": train_metrics.get("train_auc", 0)
                        }
                        
                        print(f"    Top features: {[f[0] for f in top_features[:5]]}")
                    
            except Exception as e:
                print(f"  Error: {e}")
        
        return results
    
    def generate_summary_report(self, all_results: dict) -> str:
        """Generate a comprehensive summary report."""
        report = []
        report.append("=" * 80)
        report.append("TRADING BOT ML - PROFITABILITY VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic ML Strategy Results
        if "basic_ml" in all_results:
            report.append("BASIC ML STRATEGY RESULTS:")
            report.append("-" * 40)
            
            basic_results = all_results["basic_ml"]
            profitable_strategies = 0
            total_strategies = 0
            
            for strategy_name, metrics in basic_results.items():
                total_strategies += 1
                if metrics["total_return"] > 0:
                    profitable_strategies += 1
                
                report.append(f"{strategy_name}:")
                report.append(f"  Return: {metrics['total_return']:.2f}%")
                report.append(f"  Trades: {metrics['total_trades']}")
                report.append(f"  Win Rate: {metrics['win_rate']:.1f}%")
                report.append(f"  Sharpe: {metrics['sharpe']:.2f}")
                report.append("")
            
            profitability_rate = (profitable_strategies / total_strategies * 100) if total_strategies > 0 else 0
            report.append(f"Profitability Rate: {profitability_rate:.1f}% ({profitable_strategies}/{total_strategies})")
            report.append("")
        
        # Advanced Backtesting Results
        if "advanced_backtest" in all_results:
            report.append("ADVANCED BACKTESTING RESULTS:")
            report.append("-" * 40)
            
            adv_results = all_results["advanced_backtest"]
            for test_name, metrics in adv_results.items():
                report.append(f"{test_name}:")
                report.append(f"  Annual Return: {metrics['annual_return']:.2f}%")
                report.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                report.append(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
                report.append(f"  Win Rate: {metrics['win_rate']:.1f}%")
                report.append("")
        
        # Feature Analysis Results
        if "feature_importance" in all_results:
            report.append("FEATURE ANALYSIS RESULTS:")
            report.append("-" * 40)
            
            feat_results = all_results["feature_importance"]
            for analysis_name, data in feat_results.items():
                if "top_features" in data:
                    report.append(f"{analysis_name}:")
                    report.append(f"  Total Features: {data['total_features']}")
                    report.append(f"  Train AUC: {data['train_auc']:.3f}")
                    report.append(f"  Top 5 Features: {[f[0] for f in data['top_features'][:5]]}")
                    report.append("")
        
        # Overall Assessment
        report.append("OVERALL ASSESSMENT:")
        report.append("-" * 40)
        
        # Calculate overall metrics
        if "basic_ml" in all_results:
            basic_results = all_results["basic_ml"]
            avg_return = np.mean([m["total_return"] for m in basic_results.values()])
            avg_sharpe = np.mean([m["sharpe"] for m in basic_results.values() if m["sharpe"] != 0])
            
            report.append(f"Average Return: {avg_return:.2f}%")
            report.append(f"Average Sharpe: {avg_sharpe:.2f}")
            
            if avg_return > 5 and avg_sharpe > 0.5:
                report.append("✓ SYSTEM SHOWS PROMISING PROFITABILITY")
            elif avg_return > 0:
                report.append("⚠ SYSTEM SHOWS MODEST PROFITABILITY - NEEDS OPTIMIZATION")
            else:
                report.append("✗ SYSTEM NEEDS SIGNIFICANT IMPROVEMENT")
        
        report.append("")
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        report.append("1. Focus on top-performing model types and features")
        report.append("2. Implement walk-forward optimization for parameter tuning")
        report.append("3. Use advanced risk management for better risk-adjusted returns")
        report.append("4. Consider ensemble methods for improved stability")
        report.append("5. Test on more diverse market conditions")
        
        return "\n".join(report)
    
    def run_comprehensive_test(self) -> str:
        """Run comprehensive profitability validation."""
        print("Starting comprehensive profitability validation...")
        
        # Fetch test data
        data = self.fetch_test_data(
            symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],
            timeframe="1h",
            days=180  # 6 months of data
        )
        
        if not data:
            return "ERROR: No data fetched for testing"
        
        all_results = {}
        
        # Test 1: Basic ML Strategy
        try:
            all_results["basic_ml"] = self.test_basic_ml_strategy(data)
        except Exception as e:
            print(f"Error in basic ML test: {e}")
        
        # Test 2: Advanced Backtesting
        try:
            all_results["advanced_backtest"] = self.test_advanced_backtesting(data)
        except Exception as e:
            print(f"Error in advanced backtesting: {e}")
        
        # Test 3: Walk-Forward Optimization
        try:
            all_results["walk_forward"] = self.test_walk_forward_optimization(data)
        except Exception as e:
            print(f"Error in walk-forward test: {e}")
        
        # Test 4: Feature Importance
        try:
            all_results["feature_importance"] = self.test_feature_importance(data)
        except Exception as e:
            print(f"Error in feature importance test: {e}")
        
        # Generate report
        report = self.generate_summary_report(all_results)
        
        # Save results
        self.results = all_results
        
        return report


def main():
    """Main function to run profitability validation."""
    print("Trading Bot ML - Profitability Validation")
    print("=" * 50)
    
    validator = ProfitabilityValidator()
    
    try:
        report = validator.run_comprehensive_test()
        
        # Print report
        print("\n" + report)
        
        # Save report to file
        with open("profitability_report.txt", "w") as f:
            f.write(report)
        
        print(f"\nReport saved to: profitability_report.txt")
        
        return True
        
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)