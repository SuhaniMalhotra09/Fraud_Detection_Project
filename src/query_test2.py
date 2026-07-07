import sqlite3

connection = sqlite3.connect("data/fraud_detection.db")
cursor = connection.cursor()

cursor.execute("SELECT COUNT(*) FROM transactions WHERE Class = 1")
fraud_count = cursor.fetchone()

cursor.execute("SELECT COUNT(*) FROM transactions WHERE Class = 0")
legit_count = cursor.fetchone()

print("Fraudulent transactions:", fraud_count[0])
print("Legitimate transactions:", legit_count[0])

connection.close()