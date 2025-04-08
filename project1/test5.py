import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------
# Load and Prepare Data
# ---------------------------
df = pd.read_csv("combined_data_with_champions.csv")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# ---------------------------
# Feature Selection
# ---------------------------
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
if "Champion" in numeric_features:
    numeric_features.remove("Champion")
if "Year" in numeric_features:
    numeric_features.remove("Year")

X = df[numeric_features]
y = df["Champion"]

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\nTraining set:", X_train.shape)
print("Test set:", X_test.shape)

# ---------------------------
# Model Pipelines
# ---------------------------
models = {
    "Dummy": Pipeline([("scaler", StandardScaler()), ("clf", DummyClassifier(strategy="most_frequent"))]),
    "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
    "RandomForest": Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
    "GradientBoosting": Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))]),
    "SVC": Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=42))]),
    "KNeighbors": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])
}

# ---------------------------
# Cross-Validation
# ---------------------------
cv_results = {}
print("\nCross-validation results:")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    cv_results[name] = scores.mean()
    print(f"{name}: CV Accuracy = {scores.mean():.3f}")

best_model_name = max(cv_results, key=cv_results.get)
best_model = models[best_model_name]
print(f"\nBest model based on CV accuracy: {best_model_name}")

# ---------------------------
# Fit Best Model on Training Set
# ---------------------------
best_model.fit(X_train, y_train)

# Predict hard classes
y_pred = best_model.predict(X_test)

# Evaluate with classification metrics
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# Predict Probabilities and Rank
# ---------------------------
y_probs = best_model.predict_proba(X_test)[:, 1]

# Merge identifiers for ranking
meta = df.loc[X_test.index, ["Team", "Year"]].copy()
meta["Predicted_Prob"] = y_probs
meta["Actual_Champion"] = y_test.values

# Rank within each year
meta["Predicted_Rank"] = meta.groupby("Year")["Predicted_Prob"].rank(ascending=False, method="first")

meta.to_csv("champion_predictions_ranked.csv", index=False)

# # Show where champions ranked
# champion_ranks = meta[meta["Actual_Champion"] == 1][["Year", "Team", "Predicted_Prob", "Predicted_Rank"]]
# print("\n🏆 Champion Ranks by Year:")
# print(champion_ranks.sort_values("Year"))
#
# # Top-k evaluation
# top_k = 3
# top_k_df = meta[meta["Predicted_Rank"] <= top_k]
# top_k_accuracy = top_k_df["Actual_Champion"].sum() / y_test.sum()
# print(f"\n✅ Top-{top_k} Accuracy: {top_k_accuracy:.2f}")
