import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import joblib


# ==========================================
# Load Dataset
# ==========================================

data = pd.read_csv(
    "data/student_exam_performance_dataset (1).csv"
)

print("=" * 60)
print("ORIGINAL DATA")
print("=" * 60)

print(data.head())
print("\nOriginal Shape:", data.shape)



# ==========================================
# Remove Unnecessary Columns
# ==========================================

data = data.drop(columns=[

    "student_id",
    "grade_category",

    # Remove data leakage columns
    "math_score",
    "reading_score",
    "writing_score",
    "science_score",
    "final_exam_score"

])


print("\nColumns After Cleaning:")
print(data.columns.tolist())



# ==========================================
# Encode Categorical Variables
# ==========================================

encoders = {}


categorical_columns = [

    "gender",
    "parental_education",
    "family_income",
    "internet_access",
    "study_environment",
    "tutoring",
    "pass_fail"

]


print("\nEncoding Categorical Features")


for column in categorical_columns:

    encoder = LabelEncoder()

    data[column] = encoder.fit_transform(
        data[column]
    )

    encoders[column] = encoder


    print(
        column,
        ":",
        list(
            encoder.classes_
        )
    )


# ==========================================
# Features and Target
# ==========================================


X = data.drop(
    columns=["pass_fail"]
)


y = data["pass_fail"]



print("\n")
print("=" * 60)
print("FEATURES USED FOR TRAINING")
print("=" * 60)


print(
    X.columns.tolist()
)



# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)



print("\nTraining Samples:", len(X_train))

print("Testing Samples :", len(X_test))



# ==========================================
# Save Encoders
# ==========================================


joblib.dump(

    encoders,

    "models/encoders.pkl"

)


print(
    "\nEncoders saved successfully."
)



# ==========================================
# Save Preprocessed Data
# ==========================================


joblib.dump(

    {

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test,

        "feature_names": X.columns.tolist()

    },

    "models/preprocessed_data.pkl"

)



print(
    "Preprocessed data saved successfully."
)



print("\n")
print("=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)