import numpy as np
import pandas as pd

from config import DATA_PATH, FEATURES, TARGET


def generate_demo_dataset(rows: int = 303, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic UCI-shaped demo dataset for offline development."""
    rng = np.random.default_rng(seed)
    age = rng.integers(29, 78, rows)
    sex = rng.integers(0, 2, rows)
    cp = rng.integers(0, 4, rows)
    trestbps = np.clip(rng.normal(130, 18, rows), 90, 200).round().astype(int)
    chol = np.clip(rng.normal(245, 52, rows), 125, 565).round().astype(int)
    fbs = (rng.random(rows) < 0.15).astype(int)
    restecg = rng.integers(0, 3, rows)
    thalach = np.clip(rng.normal(150, 23, rows), 70, 205).round().astype(int)
    exang = rng.integers(0, 2, rows)
    oldpeak = np.clip(rng.normal(1.0, 1.1, rows), 0, 6.2).round(1)
    slope = rng.integers(0, 3, rows)
    ca = rng.integers(0, 4, rows)
    thal = rng.integers(1, 4, rows)

    risk_score = (
        0.035 * (age - 50)
        + 0.8 * sex
        + 0.55 * (cp == 3)
        + 0.015 * (trestbps - 120)
        + 0.006 * (chol - 200)
        + 0.65 * exang
        + 0.42 * oldpeak
        + 0.5 * ca
        + 0.45 * (thal == 3)
        - 0.025 * (thalach - 140)
        + rng.normal(0, 0.85, rows)
    )
    target = (risk_score > 1.25).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
            "target": target,
        }
    )


def ensure_dataset(path=DATA_PATH) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        demo = generate_demo_dataset()
        demo.to_csv(path, index=False)
    return pd.read_csv(path)


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    df = ensure_dataset(path)
    missing_columns = [column for column in FEATURES + [TARGET] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return df[FEATURES + [TARGET]].copy()


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.replace("?", np.nan)
    for column in FEATURES + [TARGET]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in FEATURES:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    cleaned[TARGET] = cleaned[TARGET].fillna(0)
    cleaned[TARGET] = (cleaned[TARGET] > 0).astype(int)
    return cleaned.drop_duplicates().reset_index(drop=True)


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "target_counts": df[TARGET].value_counts().to_dict(),
    }

