import pandas as pd

# Load dataset
df = pd.read_csv("phishing_features.csv")

print("\n========== DATASET INFORMATION ==========")

# Number of rows and columns
print("Dataset shape:", df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())

# Data types
print("\nData types:")
print(df.dtypes)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Label distribution
print("\n========== LABEL DISTRIBUTION ==========")
print(df["label"].value_counts())

print("\nLabel percentages:")
print(df["label"].value_counts(normalize=True) * 100)

# Unique values in label
print("\nUnique labels:")
print(df["label"].unique())

# TLD information
print("\n========== TLD INFORMATION ==========")
print("Number of unique TLDs:", df["tld"].nunique())

print("\nTop 20 TLDs:")
print(df["tld"].value_counts().head(20))

# Numerical feature statistics
print("\n========== FEATURE STATISTICS ==========")
print(df.describe())

print("\n========== ANALYSIS COMPLETED ==========")