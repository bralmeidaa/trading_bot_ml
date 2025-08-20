from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


@dataclass
class SklearnClassifierWrapper:
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
