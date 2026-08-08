import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold



connection = sqlite3.connect("data/fraud_detection.db")
df = pd.read_sql_query("SELECT * FROM transaction_features", connection)
connection.close()
df = df.dropna()
X = df.drop(columns=["Class"])
Y = df["Class"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
print("Training set shape:",X_train.shape)
print("Test set shape:", X_test.shape)
print("Training fraud rate:", Y_train.mean())
print("Test fraud rate:", Y_test.mean())

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, Y_train)

train_accuracy = pipeline.score(X_train, Y_train)
test_accuracy = pipeline.score(X_test, Y_test)

print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)

Y_pred = pipeline.predict(X_test)
print(classification_report(Y_test, Y_pred))

pipeline_weighted = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

pipeline_weighted.fit(X_train, Y_train)

Y_pred_weighted = pipeline_weighted.predict(X_test)
print("=== Class Weighted Model ===")
print(classification_report(Y_test, Y_pred_weighted))
smote = SMOTE(random_state=42)
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)

print("Before SMOTE, training class counts:")
print(Y_train.value_counts())
print("After SMOTE, training class counts:")
print(Y_train_smote.value_counts())
pipeline_smote = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline_smote.fit(X_train_smote, Y_train_smote)

Y_pred_smote = pipeline_smote.predict(X_test)
print("=== SMOTE Model ===")
print(classification_report(Y_test, Y_pred_smote))

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    pipeline_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    pipeline_model.fit(X_train_smote, Y_train_smote)
    Y_pred_model = pipeline_model.predict(X_test)

    print(f"=== {name} ===")
    print(classification_report(Y_test, Y_pred_model))

    param_distributions = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [5, 10, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__class_weight": ["balanced", None]
}

rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

random_search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    scoring="recall",
    cv=3,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_smote, Y_train_smote)

print("Best parameters:", random_search.best_params_)

Y_pred_tuned_rf = random_search.predict(X_test)
print("=== Tuned Random Forest ===")
print(classification_report(Y_test, Y_pred_tuned_rf))


param_distributions_v2 = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [5, 10, 20, None],
    "model__min_samples_split": [2, 5, 10],
}

rf_pipeline_v2 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))
])

stratified_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search_v2 = RandomizedSearchCV(
    rf_pipeline_v2,
    param_distributions=param_distributions_v2,
    n_iter=10,
    scoring="recall",
    cv=stratified_cv,
    random_state=42,
    n_jobs=-1
)

random_search_v2.fit(X_train, Y_train)

print("Best parameters (v2):", random_search_v2.best_params_)

Y_pred_tuned_rf_v2 = random_search_v2.predict(X_test)
print("=== Tuned Random Forest (v2 - trained on real imbalanced data) ===")
print(classification_report(Y_test, Y_pred_tuned_rf_v2))