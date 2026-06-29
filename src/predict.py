import joblib
import pandas as pd

from config import FEATURES, MODEL_PATH


def load_model(model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            "Model file not found. Run `python -m src.train` before starting the app."
        )
    return joblib.load(model_path)


def make_prediction(input_data: dict, model=None) -> dict:
    model = model or load_model()
    row = {feature: input_data[feature] for feature in FEATURES}
    frame = pd.DataFrame([row])
    prediction = int(model.predict(frame)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = round(float(model.predict_proba(frame)[0][1]) * 100, 2)

    return {
        "prediction": prediction,
        "label": "Heart disease risk detected" if prediction == 1 else "Low heart disease risk",
        "risk_probability": probability,
    }

