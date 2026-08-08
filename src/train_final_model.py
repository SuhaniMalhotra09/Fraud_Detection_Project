import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

connection = sqlite3.connect("data/fraud_detection.db")
df = pd.read_sql_query("SELECT * FROM transaction_features", connection)
connection.close()
df = df.dropna()

X = df.drop(columns=["Class"])
Y = df["Class"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

final_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=300,
        min_samples_split=10,
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ))
])

final_pipeline.fit(X_train, Y_train)
print("Model trained.")

joblib.dump(final_pipeline, "api/fraud_model.joblib")
print("Model saved to api/fraud_model.joblib")