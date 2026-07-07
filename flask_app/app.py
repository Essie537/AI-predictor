from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# -----------------------------------
# Load the trained AI model
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "student_performance_model.pkl"

model = joblib.load(MODEL_PATH)

# -----------------------------------
# Home Page
# -----------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


# -----------------------------------
# Prediction
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    gender = request.form["gender"]

    attendance = float(request.form["attendance"])

    study_hours = float(request.form["study_hours"])

    previous_grade = float(request.form["previous_grade"])

    extracurricular = float(request.form["extracurricular"])

    parental_support = request.form["parental_support"]

    online_classes = request.form["online_classes"]

    input_data = pd.DataFrame({
        "Gender": [gender],
        "AttendanceRate": [attendance],
        "StudyHoursPerWeek": [study_hours],
        "PreviousGrade": [previous_grade],
        "ExtracurricularActivities": [extracurricular],
        "ParentalSupport": [parental_support],
        "Online Classes Taken": [online_classes]
    })

    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)