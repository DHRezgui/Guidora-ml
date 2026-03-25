"""LightGBM smoke test for environment verification.

Run: python verify_lightgbm_setup.py
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def main() -> None:
    # Small synthetic dataset to verify fit/predict end-to-end.
    x, y = make_classification(
        n_samples=600,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        random_state=42,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=30,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )
    model.fit(x_train, y_train)

    probs = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    print("LightGBM smoke test: OK")
    print(f"AUC={auc:.4f}")


if __name__ == "__main__":
    main()
