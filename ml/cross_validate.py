import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Load dataset
df = pd.read_csv("phishing_features_new.csv")

X = df.drop("label", axis=1)
y = df["label"]


# Models
models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
}


# 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)


for name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1"
        ],
        n_jobs=-1
    )

    print("\n" + name)
    print("-" * 40)

    print(
        f"Accuracy : "
        f"{scores['test_accuracy'].mean():.4f}"
    )

    print(
        f"Precision: "
        f"{scores['test_precision'].mean():.4f}"
    )

    print(
        f"Recall   : "
        f"{scores['test_recall'].mean():.4f}"
    )

    print(
        f"F1 Score : "
        f"{scores['test_f1'].mean():.4f}"
    )
    