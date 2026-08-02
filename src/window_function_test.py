import sqlite3
import pandas as pd
connection = sqlite3.connect("data/fraud_detection.db")
query = """
SELECT
    Time,
    Amount,
    Class,
    AVG(Amount) OVER (ORDER BY Time ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS rolling_avg_amount
FROM transactions
ORDER BY Time
LIMIT 20;
"""
df = pd.read_sql_query(query,connection)
print(df)
connection.close()
