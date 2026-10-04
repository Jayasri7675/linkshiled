import pandas as pd

# Load dataset
df = pd.read_csv("phishing_url_dataset_unique.csv")

print("===== DATASET OVERVIEW =====")
print("Shape:", df.shape)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== LABEL DISTRIBUTION =====")
print(df["label"].value_counts())

print("\n===== LABEL PERCENTAGE =====")
print(df["label"].value_counts(normalize=True) * 100)

print("\n===== SOURCE DISTRIBUTION =====")
print(df["source"].value_counts())

print("\n===== LABEL 0 EXAMPLES =====")
print(
    df[df["label"] == 0][["url", "label", "source"]]
    .head(10)
    .to_string(index=False)
)

print("\n===== LABEL 1 EXAMPLES =====")
print(
    df[df["label"] == 1][["url", "label", "source"]]
    .head(10)
    .to_string(index=False)
)