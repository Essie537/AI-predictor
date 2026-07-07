# Student Performance Prediction System
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Get the project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Build the path to the CSV file
DATA_PATH = BASE_DIR / "data" / "student_performance_updated_1000 .csv"

# Read the dataset
df = pd.read_csv(DATA_PATH)

# Display the first five rows
# Display the first five rows
print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

# Dataset information
print("\n========== DATASET INFORMATION ==========\n")
print(df.info())

# Statistical summary
print("\n========== STATISTICAL SUMMARY ==========\n")
print(df.describe())

# Check for missing values
print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# Check for duplicate rows
print("\n========== DUPLICATE ROWS ==========\n")
print(df.duplicated().sum())