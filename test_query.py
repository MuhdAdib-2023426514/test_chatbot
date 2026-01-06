import duckdb
import os
from datetime import datetime

# Change to the correct directory
os.chdir(r"C:\Users\Inteleon\Desktop\Adib\test_chatbot")

print("=== Current working directory ===")
print(os.getcwd())
print()

# Test 1: Direct text match
print("=== Test 1: Direct text match ===")
result1 = duckdb.sql("SELECT * FROM blood_donation_events.csv WHERE event_date = '6 January 2026' LIMIT 3")
print(result1)
print()

# Test 2: Using STRPTIME with CURRENT_DATE
print("=== Test 2: STRPTIME with CURRENT_DATE ===")
print(f"CURRENT_DATE is: {datetime.now().date()}")
result2 = duckdb.sql("SELECT CURRENT_DATE")
print(f"DuckDB CURRENT_DATE: {result2}")
result3 = duckdb.sql("SELECT * FROM blood_donation_events.csv WHERE STRPTIME(event_date, '%d %B %Y') = CURRENT_DATE LIMIT 3")
print(result3)
print()

# Test 3: Check the actual parsed dates
print("=== Test 3: Parsed dates sample ===")
result4 = duckdb.sql("SELECT event_date, STRPTIME(event_date, '%d %B %Y') as parsed_date FROM blood_donation_events.csv WHERE event_date LIKE '%January 2026%' LIMIT 5")
print(result4)

# Test 4: Test from a different directory (simulating the issue)
print("\n=== Test 4: Test from root directory ===")
os.chdir("C:\\")
try:
    result5 = duckdb.sql("SELECT COUNT(*) FROM blood_donation_events.csv")
    print(result5)
except Exception as e:
    print(f"Error: {e}")
