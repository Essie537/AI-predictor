import pandas as pd
from pathlib import Path
import joblib

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# Load Dataset
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_performance_updated_1000 .csv"

df = pd.read_csv(DATA_PATH)

# ==========================================
# Keep only the features we need

df = df[
    [
        "Gender",
        "AttendanceRate",
        "StudyHoursPerWeek",
        "PreviousGrade",
        "ExtracurricularActivities",
        "ParentalSupport",
        "Online Classes Taken",
        "FinalGrade",
    ]
]

# ==========================================
# Remove rows where FinalGrade is missing
# ==========================================

df = df.dropna(subset=["FinalGrade"])

# ==========================================
# Separate Features and Target
# ==========================================

X = df.drop("FinalGrade", axis=1)
y = df["FinalGrade"]

# ==========================================
# Identify column types
# ==========================================

numeric_features = X.select_dtypes(include=["float64", "int64"]).columns
categorical_features = X.select_dtypes(include=["object", "bool", "string"]).columns

# ==========================================
# Preprocessing
# ==========================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ==========================================
# Build Model Pipeline
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# ==========================================
# Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Train Model
# ==========================================

print("Training AI model...\n")

model.fit(X_train, y_train)

print("Training completed.\n")

# ==========================================
# Evaluate Model
# ==========================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("========== MODEL PERFORMANCE ==========")
print(f"Mean Absolute Error : {mae:.2f}")
print(f"R² Score            : {r2:.2f}")

# ==========================================
# Save Model
# ==========================================

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "student_performance_model.pkl"

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print(MODEL_PATH)