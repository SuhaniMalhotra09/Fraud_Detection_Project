import sqlite3

connection = sqlite3.connect("data/fraud_detection.db")
cursor = connection.cursor()

cursor.execute("PRAGMA table_info(transactions)")
columns = cursor.fetchall()

for col in columns:
    print(col)

connection.close()