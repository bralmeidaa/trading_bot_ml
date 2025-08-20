================================================================================
🔍 COMPREHENSIVE DIAGNOSTIC ANALYSIS REPORT
================================================================================
Generated: 2025-08-20 20:10:01

📊 ANALYSIS: BTCUSDT 1m
--------------------------------------------------
Data Quality:
  Candles: 1000
  Mean Return: 0.0005%
  Volatility: 0.05%
  Trend Changes: 98

Label Distribution:
  h1_c5: 13.3% positive
  h1_c10: 4.1% positive
  h1_c15: 1.0% positive
  h2_c10: 6.4% positive
  h3_c15: 5.4% positive
  h5_c20: 5.3% positive

Model Performance:
  xgboost: Acc 0.852, AUC 0.607
  lightgbm: Acc 0.910, AUC 0.588
  ensemble: Acc 0.893, AUC 0.574

Trading Simulation:
  Best Threshold: threshold_0.65 (0.01% return)
  Best Trades: 7


📊 ANALYSIS: BTCUSDT 5m
--------------------------------------------------
Data Quality:
  Candles: 1000
  Mean Return: -0.0035%
  Volatility: 0.10%
  Trend Changes: 132

Label Distribution:
  h1_c5: 24.8% positive
  h1_c10: 12.2% positive
  h1_c15: 5.8% positive
  h2_c10: 17.4% positive
  h3_c15: 13.3% positive
  h5_c20: 12.8% positive

Model Performance:
  xgboost: Acc 0.800, AUC 0.424
  lightgbm: Acc 0.800, AUC 0.445
  ensemble: Acc 0.800, AUC 0.651

Trading Simulation:
  Best Threshold: threshold_0.5 (0.00% return)
  Best Trades: 0


📊 ANALYSIS: ETHUSDT 5m
--------------------------------------------------
Data Quality:
  Candles: 1000
  Mean Return: -0.0039%
  Volatility: 0.22%
  Trend Changes: 110

Label Distribution:
  h1_c5: 37.4% positive
  h1_c10: 29.2% positive
  h1_c15: 21.3% positive
  h2_c10: 34.7% positive
  h3_c15: 30.3% positive
  h5_c20: 31.5% positive

Model Performance:
  xgboost: Acc 0.590, AUC 0.429
  lightgbm: Acc 0.590, AUC 0.478
  ensemble: Acc 0.590, AUC 0.505

Trading Simulation:
  Best Threshold: threshold_0.5 (0.00% return)
  Best Trades: 0


💡 KEY FINDINGS & RECOMMENDATIONS
--------------------------------------------------
🚨 IDENTIFIED ISSUES:
  - LOW PROFITABILITY: All configurations show <5% returns
  - LABEL IMBALANCE: Severe class imbalance detected

🔧 RECOMMENDATIONS:
  1. REDUCE TRANSACTION COSTS: Try lower cost_bp values (3-8)
  2. INCREASE POSITION SIZE: Test 30-50% position sizes
  3. OPTIMIZE THRESHOLDS: Try lower entry thresholds (0.52-0.58)
  4. EXTEND HOLDING PERIODS: Allow longer trades (50-100 candles)
  8. REBALANCING: Use SMOTE or class weights
  9. THRESHOLD TUNING: Adjust for imbalanced classes
  10. DIFFERENT LABELING: Try regression instead of classification

🎯 NEXT STEPS:
  1. Implement recommended changes
  2. Test with longer timeframes (15m, 1h)
  3. Try different symbols with higher volatility
  4. Consider ensemble of multiple timeframes
  5. Implement dynamic position sizing