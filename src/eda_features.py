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
plt.figure(figsize=(8, 5))
df.boxplot(column="rolling_avg_amount", by="Class")
plt.title("Rolling Average Amount: Fraud vs Legitimate")
plt.suptitle("")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Rolling Average Amount")
plt.savefig("notebooks/rolling_avg_boxplot.png")
plt.show()