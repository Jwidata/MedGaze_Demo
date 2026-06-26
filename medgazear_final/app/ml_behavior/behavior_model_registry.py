"""Behavior model registry."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def behavior_models(random_state: int = 42) -> dict[str, object]:
    models: dict[str, object] = {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        "MLPClassifier": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0005,
                batch_size=32,
                learning_rate="adaptive",
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=random_state,
            ),
        ),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state),
        "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=200, class_weight="balanced", random_state=random_state),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoostClassifier"] = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.08, eval_metric="mlogloss", random_state=random_state)
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["LightGBMClassifier"] = LGBMClassifier(n_estimators=100, random_state=random_state, verbose=-1)
    except Exception:
        pass
    return models
