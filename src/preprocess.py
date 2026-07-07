import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer

# ==========================================
# Load Dataset
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_performance_updated_1000 .csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!\n")

# Remove StudentID and Name

df = df.drop(columns=["StudentID", "Name"])

print("Columns after dropping StudentID and Name:")
print(df.columns)
# Separate numerical and categorical columns
# ==========================================

numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns

categorical_columns = df.select_dtypes(include=["object", "bool", "string"]).columns

# ==========================================
# Fill missing numerical values with median
# ==========================================

num_imputer = SimpleImputer(strategy="median")
df[numerical_columns] = num_imputer.fit_transform(df[numerical_columns])

# ==========================================
# Fill missing categorical values with mode
# ==========================================

cat_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())