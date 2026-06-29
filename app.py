from flask import Flask, jsonify, render_template, request

from config import FEATURE_DESCRIPTIONS, FEATURES
from src.predict import load_model, make_prediction

app = Flask(__name__)
model = None


FIELD_LIMITS = {
    "age": (1, 120),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (70, 230),
    "chol": (80, 650),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (50, 230),
    "exang": (0, 1),
    "oldpeak": (0, 7),
    "slope": (0, 2),
    "ca": (0, 3),
    "thal": (1, 3),
}


def get_model():
    global model
    if model is None:
        model = load_model()
    return model


def parse_form(payload) -> tuple[dict, list[str]]:
    values = {}
    errors = []
    for feature in FEATURES:
        raw_value = payload.get(feature)
        if raw_value in (None, ""):
            errors.append(f"{feature} is required.")
            continue

        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"{feature} must be numeric.")
            continue

        lower, upper = FIELD_LIMITS[feature]
        if value < lower or value > upper:
            errors.append(f"{feature} must be between {lower} and {upper}.")
        values[feature] = value

    return values, errors


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        values, errors = parse_form(request.form)
        if errors:
            return render_template(
                "predict.html",
                features=FEATURE_DESCRIPTIONS,
                values=request.form,
                errors=errors,
            ), 400

        result = make_prediction(values, get_model())
        return render_template("result.html", result=result, values=values)

    return render_template("predict.html", features=FEATURE_DESCRIPTIONS, values={}, errors=[])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    values, errors = parse_form(payload)
    if errors:
        return jsonify({"errors": errors}), 400
    return jsonify(make_prediction(values, get_model()))


@app.route("/about")
def about():
    return render_template("about.html", features=FEATURE_DESCRIPTIONS)


if __name__ == "__main__":
    app.run(debug=True)

