from flask import Flask, render_template, request, redirect

from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user
    
)

from werkzeug.security import generate_password_hash, check_password_hash

from pathlib import Path

import joblib
import pandas as pd

from flask_app.models import db, User, Prediction

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
# Load AI Model and Encoders
# =====================================================

PROJECT_DIR = BASE_DIR.parent


MODEL_PATH = (
    PROJECT_DIR /
    "models" /
    "student_performance_model.pkl"
)


ENCODER_PATH = (
    PROJECT_DIR /
    "models" /
    "encoders.pkl"
)


model = joblib.load(MODEL_PATH)

encoders = joblib.load(ENCODER_PATH)



# =====================================================
# Home
# =====================================================

@app.route("/")
def home():

    return render_template("home.html")



@app.route("/about")
@login_required
def about():

    return render_template("about.html")



@app.route("/contact")
@login_required
def contact():

    return render_template("contact.html")



# =====================================================
# Register
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


        existing_email = User.query.filter_by(
            email=email
        ).first()


        if existing_email:

            return render_template(
                "register.html",
                error="Email already registered."
            )


        existing_username = User.query.filter_by(
            username=username
        ).first()


        if existing_username:

            return render_template(
                "register.html",
                error="Username already exists."
            )


        user = User(

            fullname=fullname,

            email=email,

            username=username,

            password=password

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


        user = User.query.filter_by(
            email=email
        ).first()



        if user and check_password_hash(
            user.password,
            password
        ):

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

        grade_data=[
            55,62,68,74,80,82,85,90,94,96
        ],

        attendance_data=[
            60,65,70,75,80,85,90,95
        ],

        study_data=[
            2,5,7,8,10,12,15,18,20
        ]

    )



# =====================================================
# Prediction
# =====================================================

@app.route("/predict", methods=["GET","POST"])
@login_required
def predict():

    if request.method == "GET":

        return render_template(
            "predict.html"
        )


    # -----------------------------
    # Get user input
    # -----------------------------

    input_data = pd.DataFrame({

        "gender": [
            request.form["gender"]
        ],

        "age": [
            int(request.form["age"])
        ],

        "parental_education": [
            request.form["parental_education"]
        ],

        "family_income": [
            request.form["family_income"]
        ],

        "internet_access": [
            request.form["internet_access"]
        ],

        "study_environment": [
            request.form["study_environment"]
        ],

        "study_hours_per_day": [
            float(request.form["study_hours_per_day"])
        ],

        "attendance_rate": [
            float(request.form["attendance_rate"])
        ],

        "sleep_hours": [
            float(request.form["sleep_hours"])
        ],

        "social_media_hours": [
            float(request.form["social_media_hours"])
        ],

        "assignment_completion_rate": [
            float(request.form["assignment_completion_rate"])
        ],

        "participation_score": [
            float(request.form["participation_score"])
        ],

        "online_courses_completed": [
            int(request.form["online_courses_completed"])
        ],

        "tutoring": [
            request.form["tutoring"]
        ],

        "previous_gpa": [
            float(request.form["previous_gpa"])
        ]

    })


    # -----------------------------
    # Encode categorical features
    # -----------------------------

    categorical_columns = [

        "gender",

        "parental_education",

        "family_income",

        "internet_access",

        "study_environment",

        "tutoring"

    ]


    for column in categorical_columns:

        input_data[column] = encoders[column].transform(
            input_data[column]
        )



    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(
        input_data
    )[0]


    # Confidence percentage

    confidence = round(
        max(model.predict_proba(input_data)[0]) * 100,
        2
    )


    # Convert prediction

    if prediction == 1:

        result = "PASS"


    else:

        result = "FAIL"



    # -----------------------------
    # AI Recommendation System
    # -----------------------------


    if result == "PASS":


        if confidence >= 90:

            performance_level = "Excellent"


        elif confidence >= 75:

            performance_level = "Good"


        else:

            performance_level = "Average"



        recommendations = [

            "Maintain your current study routine.",

            "Continue attending classes regularly.",

            "Keep completing assignments on time.",

            "Continue participating actively in class."

        ]


    else:


        performance_level = "Needs Improvement"


        recommendations = [

            "Increase your daily study hours.",

            "Improve attendance rate.",

            "Complete assignments consistently.",

            "Seek academic support when needed."

        ]



    return render_template(

    "result.html",

    prediction=result,

    confidence=confidence,

    performance_level=performance_level,

    recommendations=recommendations

)
    # =====================================================
# Prediction History
# =====================================================

@app.route("/history")
@login_required
def history():

    predictions = Prediction.query.filter_by(
        user_id=current_user.id
    ).order_by(
        Prediction.created_at.desc()
    ).all()


    return render_template(
        "history.html",
        predictions=predictions
    )
    
# =====================================================
# Run Application
# =====================================================

if __name__ == "__main__":


    with app.app_context():

        db.create_all()


    app.run(debug=True)