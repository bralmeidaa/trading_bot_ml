"""
Advanced ML models for trading signal generation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    model_type: str = "xgboost"  # xgboost, lightgbm, random_forest, gradient_boosting, logistic
    calibrate: Optional[str] = "platt"  # platt, isotonic, None
    calibrate_cv: int = 3
    feature_selection: bool = True
    max_features: int = 50
    
    # XGBoost specific
    xgb_params: Dict[str, Any] = None
    
    # LightGBM specific  
    lgb_params: Dict[str, Any] = None
    
    # Random Forest specific
    rf_params: Dict[str, Any] = None
    
    # Gradient Boosting specific
    gb_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        
        if self.lgb_params is None:
            self.lgb_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
        
        if self.rf_params is None:
            self.rf_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            }
        
        if self.gb_params is None:
            self.gb_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'subsample': 0.8,
                'random_state': 42
            }


@dataclass
class AdvancedMLModel:
    """Advanced ML model wrapper with multiple algorithms."""
    config: ModelConfig
    scaler: StandardScaler
    model: Any
    feature_names: list = None
    is_fitted: bool = False
    
    @classmethod
    def create(cls, config: ModelConfig) -> "AdvancedMLModel":
        """Create a new model instance."""
        return cls(
            config=config,
            scaler=StandardScaler(),
            model=None,
            feature_names=[],
            is_fitted=False
        )
    
    def _create_base_model(self):
        """Create the base model based on configuration."""
        if self.config.model_type == "xgboost":
            return xgb.XGBClassifier(**self.config.xgb_params)
        elif self.config.model_type == "lightgbm":
            return lgb.LGBMClassifier(**self.config.lgb_params)
        elif self.config.model_type == "random_forest":
            return RandomForestClassifier(**self.config.rf_params)
        elif self.config.model_type == "gradient_boosting":
            return GradientBoostingClassifier(**self.config.gb_params)
        elif self.config.model_type == "logistic":
            return LogisticRegression(
                max_iter=500, 
                class_weight="balanced", 
                solver="liblinear",
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Fit the model and return training metrics."""
        if X.empty or len(y) == 0:
            raise ValueError("Empty training data")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Feature selection if enabled
        if self.config.feature_selection and len(self.feature_names) > self.config.max_features:
            from .features import select_features
            selected_features = select_features(X, pd.Series(y), k=self.config.max_features)
            X = X[selected_features]
            self.feature_names = selected_features
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create base model
        base_model = self._create_base_model()
        
        # Apply calibration if requested
        if self.config.calibrate is not None:
            method = "sigmoid" if self.config.calibrate.lower() in ("platt", "sigmoid") else "isotonic"
            self.model = CalibratedClassifierCV(
                base_model, 
                cv=self.config.calibrate_cv, 
                method=method
            )
        else:
            self.model = base_model
        
        # Fit the model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            "model_type": self.config.model_type,
            "n_features": len(self.feature_names),
            "n_samples": len(y),
            "class_balance": np.bincount(y).tolist(),
            "train_auc": roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.0
        }
        
        # Cross-validation score
        if len(np.unique(y)) > 1 and len(y) > 10:
            try:
                cv_scores = cross_val_score(
                    self.model, X_scaled, y, 
                    cv=min(3, len(y) // 10), 
                    scoring='roc_auc'
                )
                metrics["cv_auc_mean"] = float(np.mean(cv_scores))
                metrics["cv_auc_std"] = float(np.std(cv_scores))
            except Exception:
                metrics["cv_auc_mean"] = 0.0
                metrics["cv_auc_std"] = 0.0
        
        return metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Use only the features that were used during training
        if self.feature_names:
            available_features = [f for f in self.feature_names if f in X.columns]
            if len(available_features) != len(self.feature_names):
                missing = set(self.feature_names) - set(available_features)
                raise ValueError(f"Missing features: {missing}")
            X = X[self.feature_names]
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available."""
        if not self.is_fitted or not self.feature_names:
            return {}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                # For tree-based models
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                # For linear models
                importance = np.abs(self.model.coef_[0])
            elif hasattr(self.model, 'base_estimator') and hasattr(self.model.base_estimator, 'feature_importances_'):
                # For calibrated tree-based models
                importance = self.model.base_estimator.feature_importances_
            else:
                return {}
            
            return dict(zip(self.feature_names, importance))
        except Exception:
            return {}
    
    def save(self, filepath: str) -> None:
        """Save the model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'config': self.config,
            'scaler': self.scaler,
            'model': self.model,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> "AdvancedMLModel":
        """Load a model from disk."""
        model_data = joblib.load(filepath)
        return cls(
            config=model_data['config'],
            scaler=model_data['scaler'],
            model=model_data['model'],
            feature_names=model_data['feature_names'],
            is_fitted=model_data['is_fitted']
        )


class EnsembleModel:
    """Ensemble of multiple ML models."""
    
    def __init__(self, model_configs: list[ModelConfig], weights: Optional[list[float]] = None):
        self.model_configs = model_configs
        self.models = [AdvancedMLModel.create(config) for config in model_configs]
        self.weights = weights or [1.0] * len(model_configs)
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Fit all models in the ensemble."""
        if X.empty or len(y) == 0:
            raise ValueError("Empty training data")
        
        metrics = {"individual_models": []}
        
        for i, model in enumerate(self.models):
            try:
                model_metrics = model.fit(X, y)
                model_metrics["model_index"] = i
                metrics["individual_models"].append(model_metrics)
            except Exception as e:
                print(f"Failed to fit model {i}: {e}")
                model_metrics = {"model_index": i, "error": str(e)}
                metrics["individual_models"].append(model_metrics)
        
        self.is_fitted = True
        
        # Calculate ensemble metrics
        fitted_models = [m for m in self.models if m.is_fitted]
        metrics["n_fitted_models"] = len(fitted_models)
        
        if fitted_models:
            # Ensemble predictions on training data
            ensemble_proba = self.predict_proba(X)
            if len(ensemble_proba) > 0 and len(np.unique(y)) > 1:
                metrics["ensemble_train_auc"] = roc_auc_score(y, ensemble_proba)
        
        return metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        fitted_models = [m for m in self.models if m.is_fitted]
        if not fitted_models:
            raise ValueError("No fitted models in ensemble")
        
        predictions = []
        valid_weights = []
        
        for i, model in enumerate(self.models):
            if model.is_fitted:
                try:
                    pred = model.predict_proba(X)
                    predictions.append(pred)
                    valid_weights.append(self.weights[i])
                except Exception:
                    continue
        
        if not predictions:
            return np.zeros(len(X))
        
        # Weighted average
        predictions = np.array(predictions)
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()  # Normalize
        
        ensemble_pred = np.average(predictions, axis=0, weights=valid_weights)
        return ensemble_pred
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels using ensemble."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get the weights of each model in the ensemble."""
        return {f"model_{i}_{config.model_type}": weight 
                for i, (config, weight) in enumerate(zip(self.model_configs, self.weights))}


def create_default_ensemble() -> EnsembleModel:
    """Create a default ensemble with diverse models."""
    configs = [
        ModelConfig(model_type="xgboost", calibrate="platt"),
        ModelConfig(model_type="lightgbm", calibrate="platt"),
        ModelConfig(model_type="random_forest", calibrate="isotonic"),
        ModelConfig(model_type="logistic", calibrate="platt")
    ]
    
    # Weight tree-based models higher
    weights = [0.3, 0.3, 0.25, 0.15]
    
    return EnsembleModel(configs, weights)


def optimize_model_hyperparameters(X: pd.DataFrame, y: np.ndarray, 
                                 model_type: str = "xgboost", 
                                 n_trials: int = 50) -> Dict[str, Any]:
    """
    Optimize hyperparameters using Optuna.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna is required for hyperparameter optimization")
    
    def objective(trial):
        if model_type == "xgboost":
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'n_jobs': -1
            }
            config = ModelConfig(model_type="xgboost", xgb_params=params)
        elif model_type == "lightgbm":
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            config = ModelConfig(model_type="lightgbm", lgb_params=params)
        else:
            raise ValueError(f"Optimization not implemented for {model_type}")
        
        # Cross-validation
        model = AdvancedMLModel.create(config)
        
        # Use a subset for faster optimization
        n_samples = min(1000, len(X))
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X.iloc[indices]
        y_subset = y[indices]
        
        try:
            model.fit(X_subset, y_subset)
            X_scaled = model.scaler.transform(X_subset)
            y_pred_proba = model.model.predict_proba(X_scaled)[:, 1]
            return roc_auc_score(y_subset, y_pred_proba)
        except Exception:
            return 0.0
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'n_trials': len(study.trials)
    }