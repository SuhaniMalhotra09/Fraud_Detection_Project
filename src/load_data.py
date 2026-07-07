import pandas as pd
import sqlite3
df = pd.read_csv("data/creditcard.csv")
connection = sqlite3.connect("data/fraud_detection.db")
df.to_sql("transactions", connection, if_exists="replace", index=False)
connection.close()
print("Data loaded successfully into fraud_detection.db")