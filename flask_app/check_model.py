import joblib

model = joblib.load(
    "models/student_performance_model.pkl"
)

print("=" * 50)
print("MODEL FEATURES")
print("=" * 50)

print(model.feature_names_in_)