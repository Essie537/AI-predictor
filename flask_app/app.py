from flask import Flask, render_template, request, redirect
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
)
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path

import joblib
import pandas as pd

from models import db, User

# =====================================================
# Create Flask App
# =====================================================

app = Flask(__name__)
app.secret_key = "student_performance_secret_key"

# =====================================================
# Database Configuration
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + str(BASE_DIR / "database.db")
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

# =====================================================
# Login Manager
# =====================================================

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# =====================================================
# Load AI Model
# =====================================================

PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "student_performance_model.pkl"

model = joblib.load(MODEL_PATH)

# =====================================================
# Home Page
# =====================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


# =====================================================
# Registration
# =====================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        fullname = request.form["fullname"]
        email = request.form["email"]
        username = request.form["username"]
        password = generate_password_hash(
            request.form["password"]
        )

        # Check whether email already exists
        existing_email = User.query.filter_by(email=email).first()

        if existing_email:
            return render_template(
                "register.html",
                error="Email already registered."
            )

        # Check whether username already exists
        existing_username = User.query.filter_by(username=username).first()

        if existing_username:
            return render_template(
                "register.html",
                error="Username already exists."
            )

        user = User(
            fullname=fullname,
            email=email,
            username=username,
            password=password,
        )

        db.session.add(user)
        db.session.commit()

        return redirect("/login")

    return render_template("register.html")


# =====================================================
# Login
# =====================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):

            login_user(user)

            return redirect("/dashboard")

        return render_template(
            "login.html",
            error="Invalid email or password."
        )

    return render_template("login.html")


# =====================================================
# Logout
# =====================================================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect("/")


# =====================================================
# Dashboard
# =====================================================

@app.route("/dashboard")
@login_required
def dashboard():

    return render_template(
        "dashboard.html",
        grade_data=[55, 62, 68, 74, 80, 82, 85, 90, 94, 96],
        attendance_data=[60, 65, 70, 75, 80, 85, 90, 95],
        study_data=[2, 5, 7, 8, 10, 12, 15, 18, 20],
    )


# =====================================================
# Prediction
# =====================================================

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():

    if request.method == "GET":
        return render_template("predict.html")

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
        "Online Classes Taken": [online_classes],
    })

    prediction = round(model.predict(input_data)[0], 2)

    return render_template(
        "result.html",
        prediction=prediction,
    )


# =====================================================
# Run Application
# =====================================================

if __name__ == "__main__":

    with app.app_context():
        db.create_all()

    app.run(debug=True)