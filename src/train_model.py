import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# Load Processed Data
# ==========================================

data = joblib.load("models/preprocessed_data.pkl")
print("=" * 60)
print("FEATURES LOADED FROM PREPROCESSED DATA")
print("=" * 60)

print(data["X_train"].columns.tolist())

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# ==========================================
# Machine Learning Models
# ==========================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )

}

best_model = None
best_accuracy = 0

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

# ==========================================
# Train & Evaluate Models
# ==========================================

for name, model in models.items():

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# ==========================================
# Save Best Model
# ==========================================

joblib.dump(best_model, "models/student_performance_model.pkl")

print("\n")
print("=" * 70)
print("BEST MODEL SAVED SUCCESSFULLY")
print("=" * 70)

print(f"Best Accuracy : {best_accuracy:.4f}")

# ==========================================
# Feature Importance
# ==========================================

print("\n")
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({

        "Feature": X_train.columns,

        "Importance": best_model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(importance)

else:

    print("This model does not support feature importance.")