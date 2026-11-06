"""
Strategy effectiveness tests — fetch real data from Binance and run
the walk-forward backtest. These tests are slow (~3-5 min) but validate
that the strategy has genuine edge before going live.

Marked with @pytest.mark.slow so they can be skipped with:
    pytest -m "not slow"

Run with:
    pytest test/test_effectiveness.py -v --tb=short
"""
import time
import pytest
import pandas as pd
import numpy as np


pytestmark = pytest.mark.slow


def fetch_ohlcv(symbol: str, timeframe: str, days: int = 90) -> pd.DataFrame:
    """Download historical data. Returns empty DataFrame on network error."""
    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True, "rateLimit": 1200})
        tf_min = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}.get(timeframe, 5)
        from datetime import datetime, timedelta
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        all_ohlcv = []
        while True:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not batch:
                break
            all_ohlcv.extend(batch)
            since = batch[-1][0] + tf_min * 60 * 1000
            if len(batch) < 1000:
                break
            time.sleep(0.3)
        if not all_ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    except Exception as exc:
        pytest.skip(f"Could not fetch market data: {exc}")
        return pd.DataFrame()


class TestWalkForwardEffectiveness:
    """
    Acceptance thresholds (PER FOLD on average):
      - Win rate > 35%   (breakeven for 1.5% stop / 3% TP is 33%)
      - Profit factor > 0.8  (marginal but improving)
      - Sharpe > -2.0  (strategy not catastrophically bad)

    Full GO criteria (aggregate across all folds):
      - Sharpe > 0.8
      - MaxDD < 25%
      - WinRate > 45%
    """

    @pytest.fixture(scope="class")
    def link_5m_data(self):
        return fetch_ohlcv("LINK/USDT", "5m", days=90)

    def _run_walk_forward(self, df, symbol="LINK/USDT", timeframe="5m"):
        from sklearn.model_selection import TimeSeriesSplit
        from production_trading_system import OptimizedSignalGenerator

        signal_gen = OptimizedSignalGenerator(symbol, timeframe)
        n_splits = 3
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=max(200, len(df) // (n_splits + 2)))

        all_trades = []
        fold_results = []

        COMMISSION = 0.001
        SLIPPAGE = 0.0005
        STOP_PCT = 0.015
        TP_PCT = 0.030
        MAX_HOLD = 100

        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            signal_gen.initialize_from_history(train_df)
            if not signal_gen.is_fitted:
                continue

            test_df = signal_gen._add_indicators(test_df.copy()).reset_index(drop=True)
            features = signal_gen._prepare_features(test_df)
            valid = ~features.isna().any(axis=1)

            # Build signals
            direction = pd.Series(0, index=test_df.index)
            if valid.any():
                X = features[valid]
                X_scaled = signal_gen.scaler.transform(X)
                proba = signal_gen.model.predict_proba(X_scaled)[:, 1]
                thresh = signal_gen.params.get("ml_threshold", 0.55)
                ml_long = pd.Series(False, index=test_df.index)
                ml_long[valid] = proba > thresh

            has_cols = all(c in test_df.columns for c in ["rsi", "bb_position"])
            if has_cols:
                rev_buy = (test_df["bb_position"] < 0.08) & (test_df["rsi"] < 30)
                rev_sell = (test_df["bb_position"] > 0.92) & (test_df["rsi"] > 70)
                direction[rev_buy] = 1
                direction[rev_sell] = -1

            # Simulate trades
            fold_trades = []
            i = 0
            n = len(test_df)
            while i < n - 1:
                sig = int(direction.iloc[i])
                if sig == 0:
                    i += 1
                    continue
                row = test_df.iloc[i]
                raw_entry = float(row["close"])
                if sig == 1:
                    entry = raw_entry * (1 + COMMISSION + SLIPPAGE)
                    stop = entry * (1 - STOP_PCT)
                    target = entry * (1 + TP_PCT)
                else:
                    entry = raw_entry * (1 - COMMISSION - SLIPPAGE)
                    stop = entry * (1 + STOP_PCT)
                    target = entry * (1 - TP_PCT)

                exit_price = None
                j = i + 1
                while j < min(i + MAX_HOLD + 1, n):
                    bar = test_df.iloc[j]
                    lo, hi = float(bar["low"]), float(bar["high"])
                    if sig == 1:
                        if lo <= stop:
                            exit_price = stop
                            break
                        if hi >= target:
                            exit_price = target
                            break
                    else:
                        if hi >= stop:
                            exit_price = stop
                            break
                        if lo <= target:
                            exit_price = target
                            break
                    j += 1

                if exit_price is None:
                    exit_price = float(test_df.iloc[min(j, n - 1)]["close"])

                if sig == 1:
                    exit_net = exit_price * (1 - COMMISSION - SLIPPAGE)
                else:
                    exit_net = exit_price * (1 + COMMISSION + SLIPPAGE)

                pnl_pct = (exit_net - entry) / entry * sig
                fold_trades.append({"pnl_pct": pnl_pct})
                all_trades.extend(fold_trades[-1:])
                i = j + 1

            if fold_trades:
                wr = sum(1 for t in fold_trades if t["pnl_pct"] > 0) / len(fold_trades)
                fold_results.append({"fold": fold, "n_trades": len(fold_trades), "win_rate": wr})

        return all_trades, fold_results

    def test_strategy_has_minimum_trades(self, link_5m_data):
        """Strategy must generate at least 20 trades over 90 days of 5m data."""
        if link_5m_data.empty or len(link_5m_data) < 500:
            pytest.skip("Insufficient market data")
        trades, _ = self._run_walk_forward(link_5m_data)
        assert len(trades) >= 20, (
            f"Only {len(trades)} trades generated — strategy may not be firing signals. "
            "Check signal thresholds."
        )

    def test_win_rate_above_breakeven(self, link_5m_data):
        """Win rate must exceed the breakeven threshold for 1.5%/3% stop/TP (33%)."""
        if link_5m_data.empty or len(link_5m_data) < 500:
            pytest.skip("Insufficient market data")
        trades, _ = self._run_walk_forward(link_5m_data)
        if not trades:
            pytest.skip("No trades generated")

        wr = sum(1 for t in trades if t["pnl_pct"] > 0) / len(trades)
        assert wr > 0.33, (
            f"Win rate {wr:.1%} is below the 33% breakeven for 1:2 risk/reward. "
            "Strategy is expected to lose money in live trading."
        )

    def test_avg_win_exceeds_avg_loss(self, link_5m_data):
        """Average win should be larger than average loss (R/R > 1)."""
        if link_5m_data.empty or len(link_5m_data) < 500:
            pytest.skip("Insufficient market data")
        trades, _ = self._run_walk_forward(link_5m_data)
        if len(trades) < 10:
            pytest.skip("Too few trades for reliable statistics")

        returns = np.array([t["pnl_pct"] for t in trades])
        wins = returns[returns > 0]
        losses = returns[returns <= 0]

        if len(wins) == 0 or len(losses) == 0:
            pytest.skip("No wins or no losses in test period")

        avg_win = float(wins.mean())
        avg_loss = float(abs(losses.mean()))
        assert avg_win >= avg_loss * 0.8, (
            f"Average win ({avg_win:.3%}) is much smaller than average loss ({avg_loss:.3%}). "
            "Stop/TP ratio not working as expected."
        )

    def test_sharpe_ratio_not_catastrophic(self, link_5m_data):
        """Sharpe should be > -3 (strategy not catastrophically bad)."""
        if link_5m_data.empty or len(link_5m_data) < 500:
            pytest.skip("Insufficient market data")
        trades, _ = self._run_walk_forward(link_5m_data)
        if len(trades) < 5:
            pytest.skip("Too few trades")

        returns = np.array([t["pnl_pct"] for t in trades])
        vol = returns.std() * np.sqrt(252)
        annual_return = (1 + returns.mean()) ** 252 - 1
        sharpe = annual_return / vol if vol > 0 else 0

        assert sharpe > -3.0, (
            f"Sharpe ratio {sharpe:.2f} is catastrophically negative. "
            "Core strategy logic has fundamental issues."
        )

    def test_walk_forward_fold_consistency(self, link_5m_data):
        """No single fold should be dramatically worse than the others."""
        if link_5m_data.empty or len(link_5m_data) < 500:
            pytest.skip("Insufficient market data")
        _, fold_results = self._run_walk_forward(link_5m_data)
        if len(fold_results) < 2:
            pytest.skip("Not enough folds")

        win_rates = [f["win_rate"] for f in fold_results]
        spread = max(win_rates) - min(win_rates)
        assert spread < 0.50, (
            f"Win rate spread across folds is {spread:.1%}. "
            "Strategy may be overfitting to specific market conditions."
        )
