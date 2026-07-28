from datetime import datetime, timedelta, timezone

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


# ==========================================
# User Model
# ==========================================

class User(UserMixin, db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )


    fullname = db.Column(
        db.String(100),
        nullable=False
    )


    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )


    username = db.Column(
        db.String(50),
        unique=True,
        nullable=False
    )


    password = db.Column(
        db.String(255),
        nullable=False
    )


    # Relationship with predictions

    predictions = db.relationship(
        "Prediction",
        backref="user",
        lazy=True
    )



# ==========================================
# Prediction Model
# ==========================================

class Prediction(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    student_name = db.Column(
        db.String(100),
        nullable=False
    )

    result = db.Column(
        db.String(20),
        nullable=False
    )

    confidence = db.Column(
        db.Float,
        nullable=False
    )

    performance_level = db.Column(
        db.String(50)
    )

    attendance_rate = db.Column(
        db.Float
    )

    study_hours = db.Column(
        db.Float
    )

    previous_gpa = db.Column(
        db.Float
    )

    created_at = db.Column(
    db.DateTime,
    default=lambda: datetime.now(timezone.utc).astimezone(
        timezone(timedelta(hours=3))
    ).replace(tzinfo=None)
)


    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id")
    )