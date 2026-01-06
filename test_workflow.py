import os
import sys

# Set the correct working directory
os.chdir(r"C:\Users\Inteleon\Desktop\Adib\test_chatbot")
sys.path.insert(0, r"C:\Users\Inteleon\Desktop\Adib\test_chatbot")

from ai_agent import run_text2sql_workflow

print("Testing 'list all events today'...")
print("=" * 50)

result = run_text2sql_workflow("list all events today")

print("\n=== SQL Query ===")
print(repr(result.get("sql_query")))

print("\n=== Query Result (first 1000 chars) ===")
query_result = result.get("query_result", "None")
print(query_result[:1000] if query_result else "None")

print("\n=== Final Answer ===")
print(result.get("final_answer", "No answer"))
