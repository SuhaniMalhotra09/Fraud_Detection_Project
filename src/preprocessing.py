import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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

