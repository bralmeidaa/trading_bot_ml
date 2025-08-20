from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd

from .base import BaseStrategy
from ..backtesting.engine import BacktestResult
from ..ml.features import build_features, build_labels, select_features, create_feature_interactions
from ..ml.models import UnifiedMLModel, create_model
from ..core.risk import position_size
from ..core.advanced_risk import AdvancedRiskManager, AdvancedRiskParams
from ..utils.backtest_metrics import calc_max_drawdown, calc_sharpe


class MLProbStrategy(BaseStrategy):
    """
    ML probability strategy for short-term markets.
    - Build features from candles+indicators
    - Train a classifier (LogReg) on the first part of the window
    - Predict probabilities on the test part
    - Go long when prob > threshold and regime filters pass
    - Exit when prob < exit_threshold or opposite signal
    Costs/slippage are approximated via equity marking on fills.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        exit_threshold: float = 0.5,
        train_ratio: float = 0.7,
        risk_per_trade_pct: float = 1.0,
        min_atr_pct: float = 0.001,  # 0.1%
        max_atr_pct: float = 0.05,   # 5%
        fee_bps: float = 10.0,
        slippage_bps: float = 1.0,
        max_hold_bars: int = 60,
        # Labels
        label_horizon: int = 5,
        label_cost_bp: float = 15.0,
        # Auto threshold tuning
        auto_threshold: bool = True,
        pnl_tuning: bool = True,
        min_val_trades: int = 3,
        # ATR regime via percentiles on training set
        use_regime_percentiles: bool = True,
        regime_low_pctile: float = 20.0,
        regime_high_pctile: float = 90.0,
        # ATR-based exits
        sl_atr_mult: float = 1.2,
        tp_atr_mult: float = 2.0,
        # Probability calibration
        calibrate: str | None = None,  # 'platt' | 'isotonic' | None
        calibrate_cv: int = 3,
        # Safeguard to avoid zero signals on test
        min_test_signals: int = 5,
        test_threshold_floor: float = 0.03,
        # Advanced ML options
        model_type: str = "ensemble",  # ensemble, xgboost, lightgbm, random_forest, legacy
        use_advanced_features: bool = True,
        feature_selection: bool = True,
        max_features: int = 50,
        use_advanced_risk: bool = True,
        risk_params: AdvancedRiskParams = None,
    ) -> None:
        self.threshold = threshold
        self.exit_threshold = exit_threshold
        self.train_ratio = train_ratio
        self.risk_per_trade_pct = risk_per_trade_pct
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.max_hold_bars = max_hold_bars
        self.label_horizon = label_horizon
        self.label_cost_bp = label_cost_bp
        self.auto_threshold = auto_threshold
        self.pnl_tuning = pnl_tuning
        self.min_val_trades = int(min_val_trades)
        self.use_regime_percentiles = use_regime_percentiles
        self.regime_low_pctile = regime_low_pctile
        self.regime_high_pctile = regime_high_pctile
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.calibrate = calibrate
        self.calibrate_cv = calibrate_cv
        self.min_test_signals = int(min_test_signals)
        self.test_threshold_floor = float(test_threshold_floor)
        
        # Advanced ML options
        self.model_type = model_type
        self.use_advanced_features = use_advanced_features
        self.feature_selection = feature_selection
        self.max_features = max_features
        self.use_advanced_risk = use_advanced_risk
        self.risk_params = risk_params or AdvancedRiskParams()
        
        # Initialize advanced risk manager if enabled
        self.risk_manager = None
        if self.use_advanced_risk:
            self.risk_manager = AdvancedRiskManager(self.risk_params)

    def run(self, df: pd.DataFrame) -> BacktestResult:
        if df.empty:
            return BacktestResult(trades=[], equity_curve=[], metrics={"message": "no data"})

        # Feature engineering
        fdf = build_features(df, advanced=self.use_advanced_features)
        if fdf.empty:
            return BacktestResult(trades=[], equity_curve=[], metrics={"message": "no features"})
        
        # Add feature interactions if using advanced features
        if self.use_advanced_features:
            fdf = create_feature_interactions(fdf, max_interactions=10)

        # Labels for training
        y = build_labels(fdf, horizon=self.label_horizon, cost_bp=self.label_cost_bp)
        # Align shapes
        m = min(len(fdf), len(y))
        fdf = fdf.iloc[:m].reset_index(drop=True)
        y = y.iloc[:m].reset_index(drop=True)

        # Train/test split (walk-forward simple)
        n = len(fdf)
        split = max(int(n * self.train_ratio), 10)
        if split >= n - 5:
            split = n - 5
        X_train = fdf.iloc[:split]
        y_train = y.iloc[:split]
        X_test = fdf.iloc[split:]
        y_test = y.iloc[split:]

        # Optional: derive ATR regime thresholds from training percentiles
        if self.use_regime_percentiles and "atr_pct" in X_train.columns:
            atr_tr = X_train["atr_pct"].replace([np.inf, -np.inf], np.nan).dropna()
            if not atr_tr.empty:
                low = float(np.percentile(atr_tr, self.regime_low_pctile))
                high = float(np.percentile(atr_tr, self.regime_high_pctile))
                # Only update if sane
                if low < high and low > 0:
                    self.min_atr_pct = low
                    self.max_atr_pct = high

        # Choose numeric feature columns
        feature_cols = [
            c for c in X_train.columns
            if c not in ("ts", "open", "high", "low", "close", "volume", "dt")
        ]
        
        # Feature selection if enabled
        if self.feature_selection and len(feature_cols) > self.max_features:
            selected_features = select_features(
                X_train[feature_cols], y_train, 
                method="mutual_info", k=self.max_features
            )
            feature_cols = selected_features
        
        X_train_features = X_train[feature_cols]
        X_test_features = X_test[feature_cols]

        # Create advanced ML model
        model = create_model(
            model_type=self.model_type,
            calibrate=self.calibrate,
            calibrate_cv=self.calibrate_cv,
            feature_selection=False,  # Already done above
            max_features=self.max_features
        )
        
        if len(np.unique(y_train)) < 2:
            # Not enough class diversity; fallback to no-trade
            return BacktestResult(trades=[], equity_curve=[{"ts": int(row.ts), "equity": 1000.0} for _, row in X_test.iterrows()], metrics={"message": "no class diversity"})
        
        # Fit model and get training metrics
        train_metrics = model.fit(X_train_features, y_train.to_numpy())

        # Auto threshold tuning on validation split inside training window
        if self.auto_threshold and len(X_train) >= 30:
            # Use last 20% of train as validation
            v_start = int(len(X_train) * 0.8)
            v_start = min(max(v_start, 1), len(X_train) - 1)
            X_val = X_train.iloc[v_start:]
            y_val = y_train.iloc[v_start:]
            Xv = X_val[[c for c in feature_cols]].to_numpy(dtype=float, copy=False)
            p_val = model.predict_proba(Xv)

            if self.pnl_tuning:
                # Build a validation iterator with prices/atr/ts for PnL simulation
                v_iter = X_val.copy()
                v_iter["prob"] = p_val
                v_iter["ts"] = fdf.iloc[split - (len(X_train) - v_start): split]["ts"].values
                v_iter["close"] = fdf.iloc[split - (len(X_train) - v_start): split]["close"].values
                if "atr14" in fdf.columns:
                    v_iter["atr14"] = fdf.iloc[split - (len(X_train) - v_start): split]["atr14"].values
                if "atr_pct" in fdf.columns and "atr_pct" not in v_iter.columns:
                    v_iter["atr_pct"] = fdf.iloc[split - (len(X_train) - v_start): split]["atr_pct"].values

                def simulate(th: float, ex_th: float) -> tuple[float, int]:
                    cash = 1000.0
                    position = 0
                    qty = 0.0
                    hold = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    trade_count = 0
                    last_close = None
                    for _, r in v_iter.iterrows():
                        price = float(r["close"])
                        last_close = price
                        prob = float(r["prob"])
                        atr_pct = float(r.get("atr_pct", np.nan))
                        atr = float(r.get("atr14", np.nan)) if not np.isnan(r.get("atr14", np.nan)) else (price * atr_pct if not np.isnan(atr_pct) else np.nan)
                        if self.use_regime_percentiles:
                            regime_ok = (not np.isnan(atr_pct)) and (self.min_atr_pct <= atr_pct <= self.max_atr_pct)
                        else:
                            # Regime filter disabled: don't block entries even if atr_pct is NaN
                            regime_ok = True
                        if position == 0:
                            if regime_ok and prob >= th:
                                alloc_cash = cash * (self.risk_per_trade_pct / 100.0)
                                buy_price = price * (1.0 + (self.fee_bps + self.slippage_bps) / 10000.0)
                                if buy_price > 0 and alloc_cash > 0:
                                    qty = alloc_cash / buy_price
                                    cash -= qty * buy_price
                                    position = 1
                                    hold = 0
                                    entry_price = buy_price
                                    entry_atr = atr if not np.isnan(atr) else 0.0
                        else:
                            hold += 1
                            hit_tp = False
                            hit_sl = False
                            if entry_atr > 0:
                                hit_tp = price >= (entry_price + self.tp_atr_mult * entry_atr)
                                hit_sl = price <= (entry_price - self.sl_atr_mult * entry_atr)
                            should_exit = (not regime_ok) or prob <= ex_th or (hold >= self.max_hold_bars) or hit_tp or hit_sl
                            if should_exit:
                                sell_price = price * (1.0 - (self.fee_bps + self.slippage_bps) / 10000.0)
                                cash += qty * sell_price
                                position = 0
                                qty = 0.0
                                hold = 0
                                trade_count += 1
                    # Force close at end of validation if still in position
                    if position == 1 and last_close is not None:
                        sell_price = last_close * (1.0 - (self.fee_bps + self.slippage_bps) / 10000.0)
                        cash += qty * sell_price
                        position = 0
                        qty = 0.0
                        hold = 0
                        trade_count += 1
                    return cash, trade_count

                best_eq = -1e9
                best_th = self.threshold
                best_exit = self.exit_threshold
                candidates: list[tuple[float, float, float, int]] = []  # (eq, th, ex, trades)
                # Build data-driven threshold grid from validation probs
                qs = np.linspace(0.6, 0.99, 9)
                qvals = np.unique(np.quantile(p_val, qs))
                abs_small = np.array([0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.20, 0.25, 0.30])
                grid = np.unique(np.clip(np.concatenate([qvals, abs_small]), 0.02, 0.98))
                for th in grid:
                    ex = max(0.02, min(th - 0.02, 0.8))
                    eq, tcount = simulate(float(th), float(ex))
                    candidates.append((eq, float(th), float(ex), tcount))
                # Prefer thresholds with at least min_val_trades on validation
                valid = [c for c in candidates if c[3] >= self.min_val_trades]
                pool = valid if valid else candidates
                for eq, thv, exv, _tc in pool:
                    if eq > best_eq:
                        best_eq = eq
                        best_th = thv
                        best_exit = exv
                self.threshold = best_th
                self.exit_threshold = best_exit
            else:
                # F1-driven tuning (fallback)
                best_f1 = -1.0
                best_th = self.threshold
                best_exit = self.exit_threshold
                qs = np.linspace(0.6, 0.99, 9)
                qvals = np.unique(np.quantile(p_val, qs))
                abs_small = np.array([0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.20, 0.25, 0.30])
                grid = np.unique(np.clip(np.concatenate([qvals, abs_small]), 0.02, 0.98))
                for th in grid:
                    ex = max(0.02, min(th - 0.02, 0.8))
                    y_pred = (p_val >= th).astype(int)
                    tp = int(((y_pred == 1) & (y_val.to_numpy() == 1)).sum())
                    fp = int(((y_pred == 1) & (y_val.to_numpy() == 0)).sum())
                    fn = int(((y_pred == 0) & (y_val.to_numpy() == 1)).sum())
                    precision = tp / (tp + fp) if (tp + fp) else 0.0
                    recall = tp / (tp + fn) if (tp + fn) else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
                    if f1 > best_f1:
                        best_f1 = f1
                        best_th = float(th)
                        best_exit = float(ex)
                self.threshold = best_th
                self.exit_threshold = best_exit

        probs = model.predict_proba(Xte)

        # Ensure at least a minimum number of signals on test by relaxing threshold if needed
        signals_above = int((probs >= self.threshold).sum())
        if self.min_test_signals > 0 and signals_above < self.min_test_signals:
            # Choose a new threshold based on test distribution: highest threshold that yields >= min_test_signals
            sort_desc = np.sort(probs)[::-1]
            idx = min(self.min_test_signals - 1, len(sort_desc) - 1)
            candidate = float(sort_desc[idx]) if len(sort_desc) > 0 else self.test_threshold_floor
            new_th = max(self.test_threshold_floor, candidate)
            # Adjust exit threshold maintaining a small gap
            new_ex = max(0.02, min(new_th - 0.02, 0.8))
            self.threshold = new_th
            self.exit_threshold = new_ex
            # recompute signals_above for diagnostics
            signals_above = int((probs >= self.threshold).sum())

        # Trading simulation over test part
        start_equity = 1000.0
        cash = start_equity
        trades: List[Dict] = []
        curve: List[Dict] = []
        position = 0
        qty = 0.0
        hold = 0

        # Build an iterator with necessary fields
        iter_df = X_test.copy()
        iter_df["prob"] = probs
        iter_df["ts"] = fdf.iloc[split:]["ts"].values
        iter_df["close"] = fdf.iloc[split:]["close"].values
        # Regime filter via atr_pct
        atr_col = "atr_pct"
        if atr_col not in iter_df.columns:
            iter_df[atr_col] = fdf.iloc[split:][atr_col].values if atr_col in fdf.columns else np.nan
        # ATR absolute
        if "atr14" in fdf.columns and "atr14" not in iter_df.columns:
            iter_df["atr14"] = fdf.iloc[split:]["atr14"].values

        entry_price = 0.0
        entry_atr = 0.0
        for _, row in iter_df.iterrows():
            ts = int(row["ts"])  # type: ignore
            price = float(row["close"])  # type: ignore
            prob = float(row["prob"])  # type: ignore
            atr_pct = float(row.get("atr_pct", np.nan))
            atr = float(row.get("atr14", np.nan)) if not np.isnan(row.get("atr14", np.nan)) else (price * atr_pct if not np.isnan(atr_pct) else np.nan)

            if self.use_regime_percentiles:
                regime_ok = (not np.isnan(atr_pct)) and (self.min_atr_pct <= atr_pct <= self.max_atr_pct)
            else:
                regime_ok = True

            if position == 0:
                if regime_ok and prob >= self.threshold:
                    alloc_cash = cash * (self.risk_per_trade_pct / 100.0)
                    buy_price = price * (1.0 + (self.fee_bps + self.slippage_bps) / 10000.0)
                    if buy_price > 0 and alloc_cash > 0:
                        qty = alloc_cash / buy_price
                        cash -= qty * buy_price
                        position = 1
                        hold = 0
                        entry_price = buy_price
                        entry_atr = atr if not np.isnan(atr) else 0.0
                        trades.append({"ts": ts, "side": "BUY", "price": buy_price, "qty": qty})
            else:  # position == 1
                hold += 1
                hit_tp = False
                hit_sl = False
                if entry_atr > 0:
                    hit_tp = price >= (entry_price + self.tp_atr_mult * entry_atr)
                    hit_sl = price <= (entry_price - self.sl_atr_mult * entry_atr)
                should_exit = (not regime_ok) or prob <= self.exit_threshold or (hold >= self.max_hold_bars) or hit_tp or hit_sl
                if should_exit:
                    sell_price = price * (1.0 - (self.fee_bps + self.slippage_bps) / 10000.0)
                    cash += qty * sell_price
                    trades.append({"ts": ts, "side": "SELL", "price": sell_price, "qty": qty})
                    position = 0
                    qty = 0.0
                    hold = 0
                    entry_price = 0.0
                    entry_atr = 0.0

            mark_equity = cash + (qty * price if position == 1 else 0.0)
            curve.append({"ts": ts, "equity": mark_equity})

        # Force-close open position at the end of test period (single consolidated block)
        if position == 1 and not iter_df.empty:
            last = iter_df.iloc[-1]
            last_close = float(last["close"]) if "close" in last else None
            ts_last = int(last["ts"]) if "ts" in last else None
            if last_close is not None:
                sell_price = last_close * (1.0 - (self.fee_bps + self.slippage_bps) / 10000.0)
                cash += qty * sell_price
                trades.append({"ts": ts_last if ts_last is not None else 0, "side": "SELL", "price": sell_price, "qty": qty})
                position = 0
                qty = 0.0
                hold = 0
                # Append final equity point with timestamp
                final_equity = cash
                curve.append({"ts": ts_last if ts_last is not None else 0, "equity": final_equity})

        end_equity = cash
        pnl = end_equity - start_equity
        returns = (end_equity / start_equity) - 1.0

        # Compute paired win rate
        win_trades = 0
        total_trades = 0
        for i in range(0, len(trades), 2):
            if i + 1 < len(trades):
                buy = trades[i]
                sell = trades[i + 1]
                total_trades += 1
                if sell["price"] > buy["price"]:
                    win_trades += 1
        win_rate = (win_trades / total_trades) * 100 if total_trades else 0

        # Diagnostics
        probs_test = iter_df["prob"].to_numpy() if len(iter_df) else np.array([])
        if self.use_regime_percentiles:
            regime_ok_flags = ((iter_df.get("atr_pct", pd.Series(index=iter_df.index, dtype=float)) >= self.min_atr_pct) & (iter_df.get("atr_pct", pd.Series(index=iter_df.index, dtype=float)) <= self.max_atr_pct)).astype(int) if "atr_pct" in iter_df.columns else np.zeros(len(iter_df), dtype=int)
        else:
            # Regime filter disabled: treat all as OK
            regime_ok_flags = np.ones(len(iter_df), dtype=int)
        signals_above = int(((probs_test >= self.threshold) & (regime_ok_flags == 1)).sum()) if len(probs_test) else 0
        prob_mean = float(np.nanmean(probs_test)) if probs_test.size else 0.0
        prob_med = float(np.nanmedian(probs_test)) if probs_test.size else 0.0
        prob_p90 = float(np.nanpercentile(probs_test, 90)) if probs_test.size else 0.0
        prob_p95 = float(np.nanpercentile(probs_test, 95)) if probs_test.size else 0.0
        regime_ok_bars = int(regime_ok_flags.sum()) if len(iter_df) else 0

        metrics = {
            "start_equity": start_equity,
            "end_equity": end_equity,
            "pnl": pnl,
            "return_pct": returns * 100,
            "trades": total_trades,
            "win_rate_pct": win_rate,
            "threshold": self.threshold,
            "exit_threshold": self.exit_threshold,
            "train_ratio": self.train_ratio,
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "max_hold_bars": self.max_hold_bars,
            "max_drawdown_pct": calc_max_drawdown(curve),
            "sharpe": calc_sharpe(curve),
            # diagnostics
            "signals_above_th": signals_above,
            "regime_ok_bars": regime_ok_bars,
            "prob_mean": prob_mean,
            "prob_median": prob_med,
            "prob_p90": prob_p90,
            "prob_p95": prob_p95,
        }

        return BacktestResult(trades=trades, equity_curve=curve, metrics=metrics)
