import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
connection = sqlite3.connect("data/fraud_detection.db")
df = pd.read_sql_query("SELECT * FROM transaction_features", connection)
connection.close()
print(df.shape)
print(df.describe())

comparison = df.groupby("Class")[["transaction_count_last_10", "rolling_avg_amount", "time_since_last_transaction"]].mean()

print(comparison)
