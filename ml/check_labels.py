import pandas as pd

df = pd.read_csv("phishing_features.csv")

print("\n===== LABEL 0 EXAMPLES =====")
print(df[df["label"] == 0][["url", "label"]].head(10).to_string(index=False))

print("\n===== LABEL 1 EXAMPLES =====")
print(df[df["label"] == 1][["url", "label"]].head(10).to_string(index=False))