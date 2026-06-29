import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from config import FEATURES, MODEL_PATH, REPORTS_DIR, TARGET
from src.data import clean_dataset, dataset_summary, load_dataset
from src.preprocess import build_preprocessor


def _evaluate_model(name, model, x_test, y_test) -> dict:
    predictions = model.predict(x_test)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, predictions, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def train_models(model_path: Path = MODEL_PATH) -> dict:
    df = clean_dataset(load_dataset())
    x = df[FEATURES]
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
    }

    results = []
    trained_pipelines = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor()), ("model", estimator)]
        )
        pipeline.fit(x_train, y_train)
        trained_pipelines[name] = pipeline
        results.append(_evaluate_model(name, pipeline, x_test, y_test))

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
    }
    grid_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(random_state=42, class_weight="balanced")),
        ]
    )
    grid_search = GridSearchCV(
        grid_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    best_result = _evaluate_model("Optimized Random Forest", best_model, x_test, y_test)
    cv_scores = cross_val_score(best_model, x, y, cv=5, scoring="f1")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)

    report = {
        "dataset": dataset_summary(df),
        "model_comparison": results,
        "best_model": best_result,
        "best_params": grid_search.best_params_,
        "cross_validation_f1_scores": [round(score, 4) for score in cv_scores],
        "mean_cross_validation_f1": round(float(cv_scores.mean()), 4),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(results + [best_result]).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    return report


if __name__ == "__main__":
    training_report = train_models()
    print(json.dumps(training_report["best_model"], indent=2))

