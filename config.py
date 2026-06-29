from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heart.csv"
MODEL_PATH = BASE_DIR / "models" / "heart_disease_model.joblib"
REPORTS_DIR = BASE_DIR / "reports"

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET = "target"

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex: 1 = male, 0 = female",
    "cp": "Chest pain type: 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic",
    "trestbps": "Resting blood pressure in mm Hg",
    "chol": "Serum cholesterol in mg/dl",
    "fbs": "Fasting blood sugar greater than 120 mg/dl: 1 = true, 0 = false",
    "restecg": "Resting ECG result: 0 = normal, 1 = ST-T abnormality, 2 = probable/definite LV hypertrophy",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina: 1 = yes, 0 = no",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of peak exercise ST segment: 0 = upsloping, 1 = flat, 2 = downsloping",
    "ca": "Number of major vessels colored by fluoroscopy, 0-3",
    "thal": "Thalassemia: 1 = normal, 2 = fixed defect, 3 = reversible defect",
    "target": "Heart disease diagnosis: 1 = disease/risk, 0 = no disease",
}

