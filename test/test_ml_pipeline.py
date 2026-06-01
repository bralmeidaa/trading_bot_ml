"""
ML Pipeline tests — verify correctness of signal generation,
label creation, and model training.

Key invariants we enforce:
  1. No look-ahead bias in labels (last row must be NaN)
  2. Reasonable label distribution (not all-zero)
  3. RandomForest uses class_weight='balanced'
  4. ML signal only generates LONG direction (binary model)
  5. Model is not retrained every single call
  6. Walk-forward validation runs without crashing
"""
import numpy as np
import pandas as pd
import pytest


class TestLabelCreation:
    def test_last_row_is_nan(self, signal_gen, sample_ohlcv):
        df = signal_gen._add_indicators(sample_ohlcv.copy())
        labels = signal_gen._create_labels(df)
        assert pd.isna(labels.iloc[-1]), (
            "Last row label must be NaN — future candle does not exist at prediction time"
        )

    def test_labels_are_binary(self, signal_gen, sample_ohlcv):
        df = signal_gen._add_indicators(sample_ohlcv.copy())
        labels = signal_gen._create_labels(df).dropna()
        unique = set(labels.unique())
        assert unique.issubset({0.0, 1.0}), f"Labels must be 0/1, got {unique}"

    def test_positive_rate_is_reasonable(self, signal_gen, sample_ohlcv):
        """Between 3% and 50% of labels should be positive (avoids degenerate classifiers)."""
        df = signal_gen._add_indicators(sample_ohlcv.copy())
        labels = signal_gen._create_labels(df).dropna()
        rate = float(labels.mean())
        assert 0.03 <= rate <= 0.50, (
            f"Positive label rate {rate:.2%} is outside [3%, 50%] — "
            "likely extreme class imbalance that will break the model"
        )

    def test_uses_shift_minus_one_not_two(self, signal_gen, sample_ohlcv):
        """Labels must use shift(-1), not shift(-2), to avoid look-ahead bias."""
        df = sample_ohlcv.copy()
        # Manually compute both and compare
        df['close_shifted_1'] = df['close'].shift(-1)
        df['close_shifted_2'] = df['close'].shift(-2)

        df2 = signal_gen._add_indicators(df.copy())
        labels = signal_gen._create_labels(df2)

        # Last row NaN → shift(-1) confirmed (with shift(-2), last TWO rows would be NaN)
        assert pd.isna(labels.iloc[-1]), "Last row must be NaN"
        # Second-to-last should NOT be NaN with shift(-1) (would be NaN with shift(-2))
        assert not pd.isna(labels.iloc[-2]), (
            "Second-to-last row must NOT be NaN — indicates shift(-2) is being used"
        )


class TestModelTraining:
    def test_model_fitted_after_initialization(self, trained_signal_gen):
        assert trained_signal_gen.is_fitted, "Model must be fitted after initialize_from_history"
        assert trained_signal_gen.model is not None
        assert trained_signal_gen.scaler is not None

    def test_model_uses_balanced_class_weight(self, trained_signal_gen):
        assert trained_signal_gen.model.class_weight == 'balanced', (
            "RandomForest must use class_weight='balanced' to handle label imbalance"
        )

    def test_scaler_is_fitted(self, trained_signal_gen, sample_ohlcv):
        """Scaler must be fitted (transform must work on new data)."""
        df = trained_signal_gen._add_indicators(sample_ohlcv.copy())
        features = trained_signal_gen._prepare_features(df)
        valid = ~features.isna().any(axis=1)
        X = features[valid]
        # Should not raise
        X_scaled = trained_signal_gen.scaler.transform(X)
        assert X_scaled.shape == X.shape

    def test_retrain_counter_increments(self, signal_gen, sample_ohlcv):
        """Counter must increment each generate_signals call."""
        signal_gen.is_fitted = False
        signal_gen._retrain_counter = 0
        signal_gen.generate_signals(sample_ohlcv.copy())
        assert signal_gen._retrain_counter == 1

    def test_model_not_retrained_when_fitted_mid_interval(self, trained_signal_gen, sample_ohlcv):
        """Once fitted, model should NOT be retrained until _retrain_interval calls."""
        import unittest.mock as mock
        trained_signal_gen._retrain_counter = 1  # far from interval boundary
        original_model = trained_signal_gen.model

        with mock.patch.object(trained_signal_gen, '_update_model') as mock_update:
            trained_signal_gen.generate_signals(sample_ohlcv.copy())
            mock_update.assert_not_called()


class TestMLSignalDirection:
    def test_ml_signal_never_generates_short(self, trained_signal_gen, sample_ohlcv):
        """Binary classifier ('will price rise?') must never produce direction=-1."""
        df = trained_signal_gen._add_indicators(sample_ohlcv.copy())
        for i in range(max(0, len(df) - 30), len(df) - 1):
            result = trained_signal_gen._check_ml_signal(df.iloc[i:i+1])
            if result is not None:
                assert result['direction'] == 1, (
                    f"ML signal generated direction={result['direction']} at row {i}. "
                    "Binary 'will price rise?' model must only generate LONG signals."
                )

    def test_ml_signal_confidence_within_bounds(self, trained_signal_gen, sample_ohlcv):
        """Confidence must be in (0, 1]."""
        df = trained_signal_gen._add_indicators(sample_ohlcv.copy())
        for i in range(max(0, len(df) - 50), len(df) - 1):
            result = trained_signal_gen._check_ml_signal(df.iloc[i:i+1])
            if result is not None:
                assert 0 < result['confidence'] <= 1.0


class TestIndicators:
    def test_indicators_added(self, signal_gen, sample_ohlcv):
        df = signal_gen._add_indicators(sample_ohlcv.copy())
        required = ['sma_20', 'ema_8', 'ema_21', 'rsi', 'bb_position',
                    'atr', 'volume_ratio', 'momentum_5', 'momentum_10']
        for col in required:
            assert col in df.columns, f"Missing indicator: {col}"

    def test_rsi_within_bounds(self, signal_gen, sample_ohlcv):
        df = signal_gen._add_indicators(sample_ohlcv.copy())
        valid_rsi = df['rsi'].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all(), "RSI must be in [0, 100]"

    def test_bb_position_mostly_in_range(self, signal_gen, sample_ohlcv):
        df = signal_gen._add_indicators(sample_ohlcv.copy())
        pos = df['bb_position'].dropna()
        in_range = ((pos >= -0.5) & (pos <= 1.5)).mean()
        assert in_range >= 0.90, "BB position must be near [0,1] for most bars"


class TestWalkForwardValidation:
    def test_walk_forward_returns_dict(self, signal_gen, large_ohlcv):
        df = signal_gen._add_indicators(large_ohlcv.copy())
        result = signal_gen.walk_forward_validate(df, n_splits=3)
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'avg_accuracy' in result or 'reason' in result

    def test_walk_forward_too_few_samples_returns_invalid(self, signal_gen, sample_ohlcv):
        tiny = sample_ohlcv.iloc[:50].copy()
        df = signal_gen._add_indicators(tiny)
        result = signal_gen.walk_forward_validate(df, n_splits=5)
        assert result['valid'] is False
