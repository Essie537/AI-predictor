import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_performance_updated_1000 .csv"

df = pd.read_csv(DATA_PATH)

# Keep only useful columns
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

# Remove missing target values
df = df.dropna(subset=["FinalGrade"])

X = df.drop("FinalGrade", axis=1)
y = df["FinalGrade"]

numeric_features = X.select_dtypes(include=["float64", "int64"]).columns
categorical_features = X.select_dtypes(include=["object", "bool", "string"]).columns

preprocessor = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [("imputer", SimpleImputer(strategy="median"))]
            ),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    ),
}

print("\nModel Comparison\n")

for name, algorithm in models.items():
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", algorithm),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"{name}")
    print(f"MAE : {mae:.2f}")
    print(f"R²  : {r2:.2f}")
    print("-" * 40)