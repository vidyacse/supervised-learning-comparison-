"""Compare 5 supervised classifiers on the same dataset, split and preprocessing.

Dataset: Adult Census Income (OpenML) - predict whether income > $50K.
It has missing values, categorical features and class imbalance, so models genuinely differ.
Falls back to the (easier) breast cancer dataset if the download fails.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SEED = 42


# ---------- 1. Load data ----------
def load_data():
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml("adult", version=2, as_frame=True)
        X, y = data.data.copy(), (data.target == ">50K").astype(int)
        name = "Adult Census Income"
    except Exception as e:
        print("Could not download Adult dataset (", e, ") -> using breast cancer fallback")
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        X, y = data.data.copy(), data.target
        name = "Breast Cancer (fallback)"
    return X, y, name


X, y, name = load_data()
if len(X) > 15000:  # subsample so SVM / KNN run in reasonable time
    X, _, y, _ = train_test_split(X, y, train_size=15000, stratify=y, random_state=SEED)
print(f"Dataset: {name} | rows={len(X)} | features={X.shape[1]} | positive rate={y.mean():.2%}")

# ---------- 2. Preprocessing (fit on TRAIN only, via Pipeline) ----------
cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
if cat_cols:
    X[cat_cols] = X[cat_cols].astype(object)  # NaN stays NaN so the imputer can handle it
print(f"Missing values: {int(X.isna().sum().sum())} | numeric={len(num_cols)} | categorical={len(cat_cols)}")

transformers = [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                  ("scale", StandardScaler())]), num_cols)]
if cat_cols:
    transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                          ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                                sparse_output=False))]), cat_cols))

# ---------- 3. One split for every model ----------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
    "KNN (k=15)": KNeighborsClassifier(n_neighbors=15),
    "SVM (RBF)": SVC(kernel="rbf", C=1.0),
    "Naive Bayes": GaussianNB(),
}

# ---------- 4. Train and evaluate ----------
rows = []
for model_name, model in models.items():
    pipe = Pipeline([("prep", ColumnTransformer(transformers)), ("clf", model)])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    score = pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, "predict_proba") \
        else pipe.decision_function(X_te)
    rows.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred, zero_division=0),
        "Recall": recall_score(y_te, pred),
        "F1": f1_score(y_te, pred),
        "ROC AUC": roc_auc_score(y_te, score),
    })
    print(f"done: {model_name}")

# ---------- 5. Comparison table + chart ----------
results = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
results.index += 1
results.index.name = "Rank"
print("\n" + results.round(4).to_string())
results.round(4).to_csv("results.csv")
print("\nSaved results.csv")

ax = results.set_index("Model")[["Accuracy", "F1", "ROC AUC"]].plot.bar(figsize=(10, 5), rot=20)
ax.set_title(f"Model comparison on {name} (test set)")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
print("Saved model_comparison.png")
