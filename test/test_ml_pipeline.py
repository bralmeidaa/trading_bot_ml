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


class TestSignalCombination:
    """
    Regression tests for _combine_signals confidence aggregation.

    Bug history: confidence was a weighted SUM (sum of active weights < 1 when
    fewer than all 4 signal types fire), so the combined confidence could never
    reach the 0.65 entry threshold with the required minimum of 2 signals.
    Result: 2 days live, signals generated (conf 0.39-0.48) but ZERO trades.
    The fix makes confidence a weighted AVERAGE (divide by active weight).
    """

    def _strong(self, sig_type, direction=1, strength=1.0, confidence=0.8):
        return {"type": sig_type, "direction": direction,
                "strength": strength, "confidence": confidence}

    def test_requires_at_least_two_signals(self, signal_gen):
        assert signal_gen._combine_signals([self._strong("momentum"), None, None, None]) is None

    def test_two_confident_signals_pass_threshold(self, signal_gen):
        """momentum+ml at production confidences must now exceed 0.65."""
        mom = self._strong("momentum", confidence=0.84)
        ml = self._strong("ml", confidence=0.60)
        result = signal_gen._combine_signals([mom, ml, None, None])
        assert result is not None
        # Weighted avg: (0.84*0.3 + 0.60*0.2)/(0.3+0.2) = 0.744
        assert result["confidence"] > 0.65
        assert result["confidence"] == pytest.approx(0.744, abs=0.01)

    def test_confidence_is_average_not_sum(self, signal_gen):
        """Confidence must never exceed the max individual confidence (it's an average)."""
        mom = self._strong("momentum", confidence=0.84)
        vol = self._strong("volume", confidence=0.90)
        result = signal_gen._combine_signals([mom, None, vol, None])
        assert result is not None
        assert result["confidence"] <= 0.90 + 1e-9

    def test_two_weak_signals_stay_below_threshold(self, signal_gen):
        """Two mediocre signals (0.5 each) average to 0.5 — must not pass 0.65."""
        s1 = self._strong("momentum", confidence=0.5)
        s2 = self._strong("ml", confidence=0.5)
        result = signal_gen._combine_signals([s1, s2, None, None])
        # Either blocked by vote gate (None) or confidence below threshold
        if result is not None:
            assert result["confidence"] < 0.65


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
