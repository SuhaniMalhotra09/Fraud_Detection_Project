import sqlite3

connection = sqlite3.connect("data/fraud_detection.db")
cursor = connection.cursor()

cursor.execute("SELECT COUNT(*) FROM transactions")
result = cursor.fetchone()

print("Total rows in transactions table:", result[0])

connection.close()