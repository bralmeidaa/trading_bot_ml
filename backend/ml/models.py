from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# Import advanced models
try:
    from .advanced_models import AdvancedMLModel, EnsembleModel, ModelConfig, create_default_ensemble
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from advanced_models import AdvancedMLModel, EnsembleModel, ModelConfig, create_default_ensemble


@dataclass
class SklearnClassifierWrapper:
    """Legacy wrapper for backward compatibility."""
    scaler: StandardScaler
    clf: object  # LogisticRegression or CalibratedClassifierCV

    @classmethod
    def create(
        cls,
        calibrate: Optional[str] = None,
        calibrate_cv: int = 3,
    ) -> "SklearnClassifierWrapper":
        base = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")
        if calibrate is not None:
            method = "sigmoid" if calibrate.lower() in ("platt", "sigmoid") else "isotonic"
            clf: object = CalibratedClassifierCV(base, cv=calibrate_cv, method=method)
        else:
            clf = base
        return cls(
            scaler=StandardScaler(),
            clf=clf,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.clf.predict_proba(Xs)[:, 1]


class UnifiedMLModel:
    """
    Unified interface for both legacy and advanced ML models.
    """
    
    def __init__(self, model_type: str = "ensemble", **kwargs):
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        
        if model_type == "legacy":
            # Use legacy sklearn wrapper
            self.model = SklearnClassifierWrapper.create(
                calibrate=kwargs.get("calibrate"),
                calibrate_cv=kwargs.get("calibrate_cv", 3)
            )
        elif model_type == "ensemble":
            # Use ensemble of advanced models
            self.model = create_default_ensemble()
        else:
            # Use single advanced model
            config = ModelConfig(
                model_type=model_type,
                calibrate=kwargs.get("calibrate", "platt"),
                calibrate_cv=kwargs.get("calibrate_cv", 3),
                feature_selection=kwargs.get("feature_selection", True),
                max_features=kwargs.get("max_features", 50)
            )
            self.model = AdvancedMLModel.create(config)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> dict:
        """Fit the model and return metrics."""
        if isinstance(X, np.ndarray):
            # Convert to DataFrame for advanced models
            if self.model_type != "legacy":
                X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        if self.model_type == "legacy":
            self.model.fit(X.values if isinstance(X, pd.DataFrame) else X, y)
            self.is_fitted = True
            return {"model_type": "legacy", "n_samples": len(y)}
        else:
            metrics = self.model.fit(X, y)
            self.is_fitted = True
            return metrics
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if isinstance(X, np.ndarray) and self.model_type != "legacy":
            # Convert to DataFrame for advanced models
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        if self.model_type == "legacy":
            return self.model.predict_proba(X.values if isinstance(X, pd.DataFrame) else X)
        else:
            return self.model.predict_proba(X)
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame], threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_feature_importance(self) -> dict:
        """Get feature importance if available."""
        if self.model_type == "legacy":
            return {}
        else:
            if hasattr(self.model, 'get_feature_importance'):
                return self.model.get_feature_importance()
            else:
                return {}


def create_model(model_type: str = "ensemble", **kwargs) -> UnifiedMLModel:
    """
    Factory function to create ML models.
    
    Args:
        model_type: Type of model ("ensemble", "xgboost", "lightgbm", "random_forest", "legacy")
        **kwargs: Additional parameters for model configuration
    
    Returns:
        UnifiedMLModel instance
    """
    return UnifiedMLModel(model_type=model_type, **kwargs)
