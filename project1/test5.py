import pandas as pd
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import classifiers
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------
# Methods: Data Loading & Cleaning
# ---------------------------
# Load the dataset. The CSV contains the combined statistics along with a Champion column
df = pd.read_csv("combined_data_with_champions.csv")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# Check for missing values
print("\nMissing values in each column:\n", df.isnull().sum())

# Optional: You might want to do further cleaning here (e.g., handling missing values,
# ensuring proper data types, etc.)

# ---------------------------
# Methods: Feature Selection & Train-Test Split
# ---------------------------
# Our target variable is "Champion" (1 if the team won the championship, 0 otherwise)
# For features, we will use the numerical performance metrics.
# We exclude "Team" (string identifier) and "Year" (which may not be predictive) from features.
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
if "Champion" in numeric_features:
    numeric_features.remove("Champion")
if "Year" in numeric_features:
    numeric_features.remove("Year")

# You might also consider removing other columns if they are not relevant.
print("\nFeatures used for modeling:", numeric_features)

X = df[numeric_features]
y = df["Champion"]

# Split the data (using stratification as Champion is likely imbalanced)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# ---------------------------
# Methods: Model Training and Validation
# ---------------------------
# We will try multiple classifiers to predict Champion status.
# We'll create pipelines that first scale the data and then apply the classifier.
models = {
    "Dummy": Pipeline([("scaler", StandardScaler()),
                       ("clf", DummyClassifier(strategy="most_frequent"))]),
    "LogisticRegression": Pipeline([("scaler", StandardScaler()),
                                    ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
    "RandomForest": Pipeline([("scaler", StandardScaler()),
                              ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
    "GradientBoosting": Pipeline([("scaler", StandardScaler()),
                                  ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))]),
    "SVC": Pipeline([("scaler", StandardScaler()),
                     ("clf", SVC(probability=True, random_state=42))]),
    "KNeighbors": Pipeline([("scaler", StandardScaler()),
                            ("clf", KNeighborsClassifier())])
}

# Evaluate each model using 5-fold cross-validation on the training set
cv_results = {}
print("\nCross-validation results:")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    cv_results[name] = scores.mean()
    print(f"{name}: CV Accuracy = {scores.mean():.3f}")

# Identify the best performing model based on CV accuracy
best_model_name = max(cv_results, key=cv_results.get)
best_model = models[best_model_name]
print(f"\nBest model based on CV accuracy: {best_model_name}")

# ---------------------------
# Methods: Evaluation on Test Set
# ---------------------------
# Train the best model on the full training set
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", test_accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
