import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load historical training data
df = pd.read_csv("data/combined_data_with_champions.csv")

# Identify numeric features (excluding target and Year)
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
if "Champion" in numeric_features:
    numeric_features.remove("Champion")
if "Year" in numeric_features:
    numeric_features.remove("Year")

# Train the model pipeline
X_train = df[numeric_features]
y_train = df["Champion"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
model.fit(X_train, y_train)

coefs = model.named_steps["clf"].coef_[0]
feature_names = X_train.columns
for name, coef in zip(feature_names, coefs):
    print(f"{name}: {coef:.3f}")


# Function to rank teams from a new season
def rank_new_season(csv_path):
    df_new = pd.read_csv(csv_path)
    X_new = df_new[numeric_features]

    # Predict and normalize probabilities
    raw_probs = model.predict_proba(X_new)[:, 1]
    df_new["Predicted_Prob"] = raw_probs / raw_probs.sum()

    # Rank within the season
    df_new["Predicted_Rank"] = df_new["Predicted_Prob"].rank(ascending=False, method="first")

    # Save to file
    output_path = csv_path.replace(".csv", "_ranked.csv")
    df_new.to_csv(output_path, index=False)
    return output_path, df_new.sort_values("Predicted_Rank")[["Predicted_Prob", "Predicted_Rank", "Team"]]


# Rank teams for 1996 and 2024 seasons
rank_1996_path, ranked_1996 = rank_new_season("test/2025_merged.csv")

rank_1996_path, ranked_1996.head()