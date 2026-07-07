import sqlite3

# Connect to the database
connection = sqlite3.connect("data/fraud_detection.db")
cursor = connection.cursor()

# Average amount for fraudulent transactions
cursor.execute("SELECT AVG(Amount) FROM transactions WHERE Class = 1")
avg_fraud = cursor.fetchone()

# Average amount for legitimate transactions
cursor.execute("SELECT AVG(Amount) FROM transactions WHERE Class = 0")
avg_legit = cursor.fetchone()

# Display the results
print("Average Fraud Transaction Amount:", avg_fraud[0])
print("Average Legitimate Transaction Amount:", avg_legit[0])

# Close the connection
connection.close()