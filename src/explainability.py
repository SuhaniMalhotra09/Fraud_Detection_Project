import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

connection = sqlite3.connect("data/fraud_detection.db")
df = pd.read_sql_query("SELECT * FROM transaction_features", connection)
connection.close()
df = df.dropna()

X = df.drop(columns=["Class"])
Y = df["Class"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

best_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=300,
        min_samples_split=10,
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ))
])

best_pipeline.fit(X_train, Y_train)
print("Model trained.")

X_test_sample = X_test.sample(n=100, random_state=42)
X_test_sample_scaled = best_pipeline.named_steps["scaler"].transform(X_test_sample)

explainer = shap.TreeExplainer(best_pipeline.named_steps["model"])
shap_values = explainer.shap_values(X_test_sample_scaled)

print("SHAP values shape:", shap_values.shape)

shap_values_fraud = shap_values[:, :, 1]

shap.summary_plot(
    shap_values_fraud,
    X_test_sample,
    feature_names=X.columns.tolist(),
    show=False
)
plt.savefig("notebooks/shap_summary_plot.png", bbox_inches="tight")
plt.show()