import sqlite3
import pandas as pd

connection = sqlite3.connect("data/fraud_detection.db")

query = """
SELECT
    Time,
    Amount,
    Class,
    COUNT(*) OVER (ORDER BY Time ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) AS transaction_count_last_10,
    AVG(Amount) OVER (ORDER BY Time ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS rolling_avg_amount,
    Time - LAG(Time, 1) OVER (ORDER BY Time) AS time_since_last_transaction
FROM transactions
ORDER BY Time
LIMIT 20;
"""

df = pd.read_sql_query(query, connection)

print(df)

connection.close()