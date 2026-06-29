# Heart Disease Prediction using Machine Learning

An end-to-end AI project that predicts whether a person may be at risk of heart disease from clinical parameters. It includes dataset handling, EDA, preprocessing, model comparison, hyperparameter tuning, a saved Scikit-learn pipeline, Flask API endpoints, and a responsive web UI.

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Flask
- HTML, CSS, JavaScript
- Joblib for model persistence

## Folder Structure

```text
heart-disease-prediction/
  app.py
  config.py
  requirements.txt
  README.md
  data/
    heart.csv
  models/
    heart_disease_model.joblib
  reports/
    metrics.json
    model_comparison.csv
    *.png
  scripts/
    run_eda.py
    run_training.py
  src/
    data.py
    eda.py
    predict.py
    preprocess.py
    train.py
  static/
    css/styles.css
    js/main.js
  templates/
    base.html
    home.html
    predict.html
    result.html
    about.html
  tests/
    test_api.py
    test_model.py
```

## Dataset

The project expects the UCI Heart Disease style schema in `data/heart.csv`. If the file is missing, the training code creates a deterministic UCI-shaped demo dataset so the full project can run offline. For final portfolio use, replace `data/heart.csv` with the real UCI/Cleveland dataset using these columns:

| Column | Meaning |
| --- | --- |
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar greater than 120 mg/dl |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels |
| thal | Thalassemia result |
| target | 1 = heart disease/risk, 0 = no disease |

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run EDA

```bash
python -m src.eda
```

EDA outputs are saved in `reports/`, including statistical summaries, distribution plots, relationship plots, and a correlation heatmap.

## Train Models

```bash
python -m src.train
```

The training pipeline:

1. Loads and cleans the dataset.
2. Handles missing values.
3. Encodes categorical features.
4. Scales numeric features.
5. Splits data into train and test sets.
6. Trains Logistic Regression, Random Forest, and SVM.
7. Evaluates accuracy, precision, recall, F1-score, and confusion matrix.
8. Tunes Random Forest with GridSearchCV.
9. Runs cross-validation.
10. Saves the final model to `models/heart_disease_model.joblib`.

## Run Flask App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## API Usage

Endpoint:

```text
POST /api/predict
```

Example JSON:

```json
{
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 130,
  "chol": 240,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.2,
  "slope": 1,
  "ca": 0,
  "thal": 2
}
```

## Testing

```bash
pytest
```

Tests cover dataset cleaning and API input validation.

## Deployment

For Flask deployment:

1. Train the model locally or in the deployment build step.
2. Ensure `models/heart_disease_model.joblib` exists.
3. Set the start command to:

```bash
gunicorn app:app
```

Add `gunicorn` to `requirements.txt` if your deployment platform requires it.

## GitHub Setup

```bash
git init
git add .
git commit -m "Build heart disease prediction ML app"
git branch -M main
git remote add origin https://github.com/<username>/heart-disease-prediction.git
git push -u origin main
```

## Screenshots

Add screenshots of:

- Home page
- Prediction form
- Result page
- EDA plots from `reports/`

## Future Improvements

- Use the full original UCI dataset directly from a data registry.
- Add SHAP explanations for model interpretability.
- Add authentication for clinical users.
- Add database storage for prediction history.
- Package the app with Docker.
- Add CI workflow for tests and linting.

## Resume Description

Built an end-to-end Heart Disease Prediction system using Python, Scikit-learn, Flask, and a responsive web interface. Implemented data cleaning, EDA, preprocessing pipelines, model comparison, GridSearchCV optimization, cross-validation, API prediction endpoints, and production-ready model serialization.

## Interview Explanation

This project solves a binary classification problem. I used clinical parameters such as age, cholesterol, blood pressure, chest pain type, ECG results, maximum heart rate, exercise-induced angina, and thalassemia. The pipeline cleans the data, imputes missing values, scales numeric values, one-hot encodes categorical values, and compares Logistic Regression, Random Forest, and SVM. I selected the best model after evaluation and GridSearchCV tuning, saved it with Joblib, and served predictions through Flask as both a web form and JSON API.

