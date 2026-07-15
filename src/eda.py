import pandas as pd

# ==========================================
# Load Dataset
# ==========================================

data = pd.read_csv("data/student_exam_performance_dataset (1).csv")

print("=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)
print(data.head())

print("\n")

print("=" * 70)
print("DATASET SHAPE")
print("=" * 70)
print(data.shape)

print("\n")

print("=" * 70)
print("COLUMN NAMES")
print("=" * 70)
print(data.columns)

print("\n")

print("=" * 70)
print("DATA TYPES")
print("=" * 70)
print(data.info())

print("\n")

print("=" * 70)
print("MISSING VALUES")
print("=" * 70)
print(data.isnull().sum())

print("\n")

print("=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)
print(data.duplicated().sum())

print("\n")

print("=" * 70)
print("PASS / FAIL DISTRIBUTION")
print("=" * 70)
print(data["pass_fail"].value_counts())

print("\n")

print("=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(data.describe())