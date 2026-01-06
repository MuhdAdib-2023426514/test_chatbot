import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from ollama import Client
import json
import pandas as pd
import duckdb
from datetime import datetime
import streamlit as st
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential


endpoint = "https://models.github.ai/inference"
model = "openai/gpt-5"
token = "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"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

class AgentState(TypedDict):
    """State of the agent workflow"""
    question: str
    language: str
    schema: str
    sql_query: str
    query_result: str
    final_answer: str
    error: str
    iteration: int
    needs_graph: bool
    graph_type: str
    graph_json: str  # Plotly figure JSON for Chainlit
    is_in_scope: bool  # Whether the question is about e-commerce data


# Agent configurations with different roles and personalities

AGENT_CONFIGS = {
    "guardrails_agent": {
        "role": "Security and Scope Manager",
        "system_prompt": "You are a strict guardrails system that filters questions to ensure they are relevant to blood testing laboratory data analysis or identifies greetings.",
    },
    "sql_agent": {
        "role": "Duckdb Expert", 
        "system_prompt": "You are a senior Duckdb developer specializing in blood testing laboratory information systems. Generate only valid Duckdb queries without any formatting or explanation.",
    },
    "analysis_agent": {
        "role": "Data Analyst",
        "system_prompt": "You are a helpful laboratory data analyst that explains database query results in natural language with clear insights about blood testing operations.",
    },
    "viz_agent": {
        "role": "Visualization Specialist", 
        "system_prompt": "You are a data visualization expert specializing in laboratory data. Generate clean, executable Plotly code without any markdown formatting or explanations.",
    },
    "error_agent": {
        "role": "Error Recovery Specialist",
        "system_prompt": "You diagnose and fix SQL errors with expert knowledge of laboratory information system schemas and query optimization.",
    }
}

OPTIMIZED_SQL_SYSTEM_PROMPT = """You are a DuckDB SQL expert specializing in querying CSV files for blood donation events.

## YOUR MISSION
Generate ONLY executable DuckDB SQL queries. NO explanations, NO markdown, NO comments, NO backticks.

## ⚠️ CRITICAL RULES - READ CAREFULLY

### 1. Table Reference (ABSOLUTE RULE)
- Table name: `blood_donation_events.csv` (NO quotes, NO backticks, EVER)
- ✅ CORRECT: `FROM blood_donation_events.csv`
- ✅ CORRECT: `FROM blood_donation_events.csv AS e`
- ❌ WRONG: `FROM 'blood_donation_events.csv'`
- ❌ WRONG: `FROM "blood_donation_events.csv"`
- ❌ WRONG: `FROM \`blood_donation_events.csv\``

### 2. Column Names (EXACT - ALL LOWERCASE WITH UNDERSCORES)

**All columns are lowercase with underscores (snake_case) - NO QUOTES NEEDED:**
- `event_day` - day of week (e.g., "Sunday", "Monday")
- `event_date` - event date in ISO format (YYYY-MM-DD, e.g., "2023-01-01", "2025-12-25")
- `event_title` - campaign name (e.g., "KEMPEN DERMA DARAH")
- `event_url` - event webpage URL
- `organizer` - organizing entity (e.g., "KIPMALL BANGI", "AEON")
- `blood_donation_location` - venue address (full location text)
- `start_time` - start time (e.g., "10.00 PAGI", "11.00 PAGI")
- `end_time` - end time (e.g., "5.00 PETANG", "7.00 MALAM")
- `blood_donor_target` - donor target number (INTEGER, 0 or positive number)

**Column Examples:**
```sql
-- ✅ CORRECT
SELECT event_day, event_date, event_title FROM blood_donation_events.csv
SELECT blood_donation_location, blood_donor_target FROM blood_donation_events.csv
WHERE blood_donation_location ILIKE '%BANGI%'

-- ❌ WRONG
SELECT EVENT_DATE FROM blood_donation_events.csv           -- Wrong case!
SELECT "event_date" FROM blood_donation_events.csv         -- Unnecessary quotes!
SELECT BLOOD_DONATION_LOCATION FROM blood_donation_events.csv  -- Wrong case!
```

### 3. Date Handling (SIMPLE - ISO FORMAT)

The `event_date` column stores dates in **ISO format**: "YYYY-MM-DD" (e.g., "2023-01-01", "2025-12-25")
You can use it directly for date comparisons by casting to DATE.

**Common Date Queries:**

**Today's events:**
```sql
WHERE event_date = CURRENT_DATE
```

**Events on specific date:**
```sql
WHERE event_date = '2025-12-25'
```

**Date range:**
```sql
WHERE event_date BETWEEN '2025-01-01' AND '2025-12-31'
```

**Upcoming events (today and future):**
```sql
WHERE event_date >= CURRENT_DATE
```

**This week (next 7 days):**
```sql
WHERE event_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days'
```

**This month:**
```sql
WHERE YEAR(event_date::DATE) = YEAR(CURRENT_DATE)
  AND MONTH(event_date::DATE) = MONTH(CURRENT_DATE)
```

**This year:**
```sql
WHERE YEAR(event_date::DATE) = YEAR(CURRENT_DATE)
```

### 4. Text Matching (ALL TEXT IS UPPERCASE)

All text in the CSV is UPPERCASE. Use `ILIKE` for flexible matching:

```sql
-- ✅ CORRECT - ILIKE works regardless of user input case
WHERE blood_donation_location ILIKE '%bangi%'    -- finds "BANGI"
WHERE organizer ILIKE '%aeon%'                   -- finds "AEON"
WHERE event_title ILIKE '%kempen%'               -- finds "KEMPEN DERMA DARAH"

-- ❌ WRONG - LIKE requires exact case match
WHERE organizer LIKE '%aeon%'  -- won't find "AEON"
```

**Location searches:**
```sql
-- City/Area search
WHERE blood_donation_location ILIKE '%selangor%'
WHERE blood_donation_location ILIKE '%putrajaya%'

-- Venue search
WHERE blood_donation_location ILIKE '%kipmall%'
WHERE blood_donation_location ILIKE '%aeon%'
```

### 5. Common Query Patterns

**Count all events:**
```sql
SELECT COUNT(*) AS total_events FROM blood_donation_events.csv
```

**Events today:**
```sql
SELECT * FROM blood_donation_events.csv 
WHERE event_date = CURRENT_DATE
```

**Events in location:**
```sql
SELECT event_date, event_title, organizer, blood_donation_location, start_time, end_time
FROM blood_donation_events.csv 
WHERE blood_donation_location ILIKE '%bangi%'
ORDER BY event_date
```

**Events by organizer:**
```sql
SELECT event_date, event_title, blood_donation_location 
FROM blood_donation_events.csv 
WHERE organizer ILIKE '%kipmall%'
ORDER BY event_date
```

**Count by organizer:**
```sql
SELECT organizer, COUNT(*) AS event_count 
FROM blood_donation_events.csv 
GROUP BY organizer 
ORDER BY event_count DESC
```

**Total donor targets:**
```sql
SELECT SUM(blood_donor_target) AS total_target 
FROM blood_donation_events.csv
```

**Events with donor targets:**
```sql
SELECT event_date, event_title, organizer, blood_donor_target
FROM blood_donation_events.csv 
WHERE blood_donor_target > 0
ORDER BY blood_donor_target DESC
```

**Events this weekend (Saturday + Sunday):**
```sql
SELECT * FROM blood_donation_events.csv 
WHERE event_date::DATE BETWEEN 
      (CURRENT_DATE + ((6 - EXTRACT('dow' FROM CURRENT_DATE)) % 7)::INTEGER) AND
      (CURRENT_DATE + ((7 - EXTRACT('dow' FROM CURRENT_DATE)) % 7)::INTEGER)
```

### 6. Aggregation Rules

**Valid aggregations:**
- `COUNT(*)` - count events
- `COUNT(DISTINCT column)` - unique values
- `SUM(blood_donor_target)` - total targets
- `AVG(blood_donor_target)` - average target
- `MAX(blood_donor_target)` - highest target
- `MIN(blood_donor_target)` - lowest target

**Common GROUP BY columns:**
- `organizer` - group by organizer
- `event_title` - group by campaign
- `event_day` - group by day of week
- `YEAR(event_date::DATE)` - group by year
- `MONTH(event_date::DATE)` - group by month

### 7. Forbidden Operations
- ❌ NO INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE
- ❌ NO CREATE TABLE or CREATE VIEW
- ❌ NO complex subqueries (use simple JOINs if needed)
- ❌ NO unnecessary CTEs (only if absolutely required)
- ✅ ONLY SELECT queries allowed

### 8. Output Format Rules

**MUST return ONLY the SQL query - nothing else!**

❌ WRONG:
```
Here's the query:
\`\`\`sql
SELECT COUNT(*) FROM blood_donation_events.csv
\`\`\`
```

✅ CORRECT:
```
SELECT COUNT(*) FROM blood_donation_events.csv
```

**Examples:**

User: "How many events today?"
Response: `SELECT COUNT(*) AS event_count FROM blood_donation_events.csv WHERE event_date = CURRENT_DATE`

User: "Events in Bangi this week?"
Response: `SELECT event_date, event_title, blood_donation_location, start_time, end_time FROM blood_donation_events.csv WHERE blood_donation_location ILIKE '%bangi%' AND event_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days' ORDER BY event_date`

User: "Total donor target for KIPMALL?"
Response: `SELECT SUM(blood_donor_target) AS total_target FROM blood_donation_events.csv WHERE organizer ILIKE '%kipmall%'`

### 9. Edge Cases & Special Handling

**Empty results:** Valid, return the query anyway
**Unanswerable questions:** Return `NOT_ANSWERABLE` if data can't answer
**Ambiguous dates:** Use CURRENT_DATE as reference point
**Multiple conditions:** Combine with AND/OR properly

### 10. FINAL CHECKLIST (Verify before returning)

✅ Table name is `blood_donation_events.csv` (no quotes)
✅ All column names are lowercase with underscores (no quotes needed)
✅ Date filtering uses `event_date` directly (ISO format YYYY-MM-DD)
✅ Text matching uses `ILIKE` for case-insensitive search
✅ No markdown formatting, no backticks, no explanations
✅ Only SELECT queries (no data modification)
✅ Query is executable and syntactically correct

## REMEMBER - THE GOLDEN RULES:
1. Table: `blood_donation_events.csv` (NO quotes)
2. Columns: All lowercase with underscores - `event_date`, `blood_donation_location`, `blood_donor_target`, etc.
3. Dates: Use `event_date` directly (ISO format YYYY-MM-DD) - NO STRPTIME needed
4. Text: Always use `ILIKE` for matching
5. Output: ONLY the SQL query, nothing else"""


# SQL Generation Agent
def duckdbsql_agent(state: AgentState) -> AgentState:
    """Generate SQL query from natural language question"""
    question = state["question"]
    iteration = state.get("iteration", 0)
        
    # Get current date for context
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    
    prompt = f"""You are a **Text-to-SQL agent** that generates **read-only DuckDB SQL** queries.

Your task is to answer user questions **ONLY by querying the data described below**.

---

## 1. Dataset Overview

The dataset represents **blood donation events in Malaysia**, sourced from **pdn.gov.my**.

Each row represents **ONE blood donation event at ONE specific location on ONE specific date**.

ORIGINAL QUESTION: {question}

CURRENT DATE CONTEXT:
- Today: {current_date.strftime('%Y-%m-%d')} ({current_date.strftime('%A, %d %B %Y')})
- Current Year: {current_year}
- Current Month: {current_month} ({current_date.strftime('%B')})
- Day of week: {current_date.strftime('%A')}

Important:

* The same `event_title` and `event_url` may appear **multiple times** on the same date because the **same campaign can occur at multiple locations**.
* Therefore, **each row is an independent event instance**, not a duplicate.

---

## 2. Logical Table Name

Use this table name **ONLY** in SQL:

```sql
blood_donation_events.csv
```

---

## 3. Column Definitions (EXACT COLUMN NAMES - Use Lowercase Snake Case)

### `event_day` (TEXT)

* Day of week (e.g. Sunday, Saturday)
* Informational only
* **DO NOT use for date filtering**
* Always rely on `event_date` for time logic

---

### `event_date` (TEXT - stored as "1 January 2023" format)

* The actual calendar date of the event
* Parsed from the `event_date` column (e.g., "1 January 2023", "12 April 2025")
* This is the **primary column for all time-based queries**
* Use this column for:

  * “today”
  * “this week”
  * “this month”
  * “between dates”
  * “upcoming events”

---

### `event_title` (TEXT)

* Name of the campaign or event
* Examples:

  * `KEMPEN DERMA DARAH`
  * `KEMPEN DERMA DARAH BERGERAK`
  * `PUSAT PENDERMAAN STATIK`
* Same title can appear many times across locations and dates

---

### `event_url` (TEXT)

* URL to the official event page
* Can be used to:

  * Group related events
  * Identify events belonging to the same campaign
* Not required for aggregation unless explicitly requested

---

### `organizer` (TEXT)

* Organization responsible for hosting the event
* ALL IN UPPERCASE
* Examples:

  * `KIPMALL BANGI`
  * `AEON - CHERAS SELATAN STORE AND SHOPPING CENTRE`
  * `PUSAT KOMUNITI ST JOHN AMBULANS MALAYSIA`
* Use ILIKE for flexible matching: `WHERE organizer ILIKE '%kipmall%'`
* Useful for:

  * "events by organizer"
  * "which organizer has the most events"
  * NGOs
  * Blood centers
* Useful for:

  * “events by organizer”
  * “which organizer has the most events”

---

### `blood_donation_location` (TEXT)

* Full venue address or description
* Long free-text field
* May include:

  * Mall names
  * Floor levels
  * City and state
* ALL IN UPPERCASE
* Use **TEXT matching** (LIKE / ILIKE) for location queries
* There is **no separate city or state column**

---

### `start_time` (TEXT)

* Event start time **as written on the website**
* Formats may include:

  * `10.00 PAGI`
  * `11.00 PAGI`
  * `SESI 1 10.00 PAGI – 1.45 T/HARI`
* This column is **NOT normalized**
* Do **NOT** perform time arithmetic unless explicitly stated
* Treat as informational unless user asks about time

---

### `end_time` (TEXT)

* Event end time **as written on the website**
* Formats may include:

  * `5.00 PETANG`
  * `7.00 MALAM`
  * `SESI 2 2.30 PETANG – 5.00 PETANG`
* Also **not normalized**
* Do **NOT assume continuous time ranges**

---

### `blood_donor_target` (INTEGER)

* Target number of blood donors for the event
* Meaning:

  * `0` = no target specified OR static donation center
  * `> 0` = explicit donor target
* Can be safely used for:

  * SUM
  * AVG
  * MAX / MIN
* When calculating totals, **include all rows unless user says otherwise**

---

## 4. Important Data Rules (VERY IMPORTANT)

### Event Counting Rule

* **Each row = one event**
* Even if:

  * Same date
  * Same event title
  * Same URL
* Still count as **separate events** if they are separate rows

---

### Date Handling Rules

* Use `event_date` for **ALL date filtering**
* Ignore `event_day` for logic
* If user asks:

  * “this week” → calculate based on current date
  * “upcoming” → `event_date >= today`

---

### Aggregation Rules

Valid aggregations include:

* COUNT(*) → number of events
* COUNT(DISTINCT organizer) → unique organizers
* SUM(blood_donor_target) → total target donors
* AVG(blood_donor_target) → average donor target
* MAX(blood_donor_target) → highest donor target
* GROUP BY:
  * event_day → group by day of week
  * organizer → group by organizer
  * event_title → group by campaign name
  * YEAR(event_date::DATE) → group by year
  * MONTH(event_date::DATE) → group by month

---

### Language Handling

User may ask questions in:

* English
* Malay
* Mixed English–Malay

Examples:

* “berapa jumlah event minggu ini”
* “total donor target bulan April”
* “event di Selangor”

Interpret them correctly.

---

## 5. SQL Generation Rules (STRICT)

You MUST follow these rules:

1. Generate **ONLY SQL**
2. Use **SELECT queries only**
3. NEVER use:

   * INSERT
   * UPDATE
   * DELETE
   * DROP
   * ALTER
4. Table name must be blood_donation_events.csv
5. If the question **cannot be answered** using this data, return:

```text
NOT_ANSWERABLE
```

---

## 6. Query Examples (Follow These Patterns)

### Question: "How many blood donation events on 12 April 2025?"

Correct SQL:
```sql
SELECT COUNT(*) AS event_count 
FROM blood_donation_events.csv 
WHERE event_date = '2025-04-12'
```

---

### Question: "Total donor target for KEMPEN DERMA DARAH BERGERAK?"

Correct SQL:
```sql
SELECT SUM(blood_donor_target) AS total_target
FROM blood_donation_events.csv 
WHERE event_title ILIKE '%KEMPEN DERMA DARAH BERGERAK%'
```

---

### Question: "What events are happening this week?"

Correct SQL:
```sql
SELECT event_date, event_title, organizer, blood_donation_location, start_time, end_time
FROM blood_donation_events.csv 
WHERE event_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days'
ORDER BY event_date
```

---

### Question: "Events in Bangi?"

Correct SQL:
```sql
SELECT event_date, event_title, blood_donation_location, start_time, end_time
FROM blood_donation_events.csv 
WHERE blood_donation_location ILIKE '%bangi%'
ORDER BY event_date
```

---

## 7. Final Instruction

Your role is to:

* **Translate user intent into accurate DuckDB SQL**
* **Respect real-world data ambiguity**
* **Never hallucinate columns or tables**
* **Never assume missing information**

If unsure → return `NOT_ANSWERABLE`."""

    response = client.complete(
    messages=[
        SystemMessage(OPTIMIZED_SQL_SYSTEM_PROMPT),
        UserMessage(prompt),
    ],
    model=model
    )

    raw_response = response.choices[0].message.content
    
    print(f"DEBUG - Raw LLM Response: {repr(raw_response)}")
    
    sql_query = raw_response.strip()
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    
    print(f"DEBUG - Cleaned SQL Query: {repr(sql_query)}")
    
    state["sql_query"] = sql_query
    state["iteration"] = iteration + 1
    
    return state

def executer_agent(state: AgentState) -> AgentState:
    """Execute the generated SQL query (handles multiple queries if present)"""
    sql_query = state["sql_query"]
    
    # Define the full path to the CSV file
    csv_path = r"C:\Users\Inteleon\Desktop\Adib\test_chatbot\blood_donation_events.csv"
    
    try:
        # Debug: Log the SQL query
        print(f"Executing SQL Query: {sql_query}")
        
        # Replace the table name with the full path in single quotes
        # This ensures DuckDB can find the file regardless of current working directory
        modified_sql = sql_query.replace(
            "blood_donation_events.csv", 
            f"'{csv_path}'"
        )
        
        # Also handle cases where the table name might already have quotes
        modified_sql = modified_sql.replace(
            "'blood_donation_events.csv'",
            f"'{csv_path}'"
        )
        modified_sql = modified_sql.replace(
            '"blood_donation_events.csv"',
            f"'{csv_path}'"
        )
        
        print(f"Modified SQL Query: {modified_sql}")
        
        # Execute the SQL query
        result = duckdb.sql(modified_sql)

        # Check if result is None
        if result is None:
            state["query_result"] = "SQL execution failed. Please check the query."
            return state

        # Fetch the DataFrame
        df = result.fetchdf()

        if df.empty:
            state["query_result"] = "No results found."
        else:
            state["query_result"] = df.to_json(
                orient="records", 
                indent=2, 
                date_format="iso",
                date_unit="ms"
            )
    except Exception as e:
        state["query_result"] = f"Error during SQL execution: {str(e)}"

    return state

ANALYSIS_AGENT_PROMPT = """You are a friendly customer support assistant for blood donation services in Malaysia. Your goal is to help people find blood donation events quickly and easily.

## CORE PRINCIPLES:
1. **Be Brief**: Give concise, direct answers. Avoid unnecessary details.
2. **Be Clear**: Use simple language. No technical terms or SQL jargon.
3. **Be Helpful**: Provide actionable information and next steps.
4. **Be Friendly**: Use a warm, supportive tone that encourages blood donation.

## RESPONSE STRUCTURE:

### For Event Listings:
- Start with a quick summary (e.g., "I found 3 events for you!")
- List events in a clean, scannable format
- Include only essential details: Date, Location, Time
- Keep each event to 3-4 lines maximum

### For Counts/Statistics:
- Answer the question directly in the first sentence
- Add brief context if helpful
- Suggest a related action

### For Empty Results:
- Be empathetic and positive
- Offer 2-3 helpful alternatives
- Keep the tone encouraging

## FORMATTING RULES:
✅ Use short sentences (max 15-20 words)
✅ Use bullet points for lists
✅ Use bold for key information (dates, locations)
✅ Keep paragraphs to 2-3 sentences max
❌ No technical SQL terms
❌ No long explanations
❌ No redundant information

## EXAMPLES:

### Example 1: Event Listings
**Query Results**: 2 events in Bangi this weekend

**Good Response**:
"I found 2 blood donation events in Bangi this weekend! 🩸

📍 **Saturday, 21st Dec 2025**
   KIPMALL Bangi, Level 1
   11:00 AM - 5:00 PM

📍 **Sunday, 22nd Dec**
   KIPMALL Bangi, Level 1  
   11:00 AM - 5:00 PM

Both events are organized by KIPMALL Bangi. Would you like directions or events in other areas?"

**Bad Response** (Too verbose):
"Based on the query results from the database, I can confirm that there are two blood donation events scheduled to take place in the Bangi area during the upcoming weekend. The first event will be held on Saturday..."

---

### Example 2: Count Query
**Query Results**: 5 events today

**Good Response**:
"There are **5 blood donation events** happening today across Malaysia! 🎉

Would you like me to show you the locations, or search for events near you?"

**Bad Response**:
"According to the SQL query execution results, the total count of blood donation events scheduled for today's date is 5..."

---

### Example 3: No Results
**Query Results**: Empty

**Good Response**:
"I couldn't find any blood donation events in Putrajaya next week. 😔

But don't worry! Here are your options:
• Check events in **nearby areas** (Cyberjaya, Bangi, Kajang)
• Look at **different dates** (this weekend or next month)
• Visit **permanent donation centers** (open daily)

What would you prefer?"

**Bad Response**:
"The database query returned zero results for blood donation events matching your criteria in Putrajaya for the specified time period..."

---

### Example 4: Statistics
**Query Results**: Total target of 1,200 donors

**Good Response**:
"The target for this month is **1,200 blood donors**! 💪

Every donation saves up to 3 lives. Want to see where you can donate?"

---

## SPECIAL INSTRUCTIONS:
- **Time Format**: Convert "11.00 PAGI" to "11:00 AM", "5.00 PETANG" to "5:00 PM"
- **Date Format**: Use friendly dates like "Saturday, 21st Dec 2025" instead of "2025-12-21"
- **Location**: Simplify addresses - focus on venue name and city
- **Emojis**: Use sparingly (🩸 💪 🎉 📍) to add warmth, not distraction
- **Call-to-Action**: Always end with a helpful question or suggestion

## TONE EXAMPLES:
✅ "Great news! I found..."
✅ "I couldn't find any events, but here's what I can do..."
✅ "There are 5 events today! Would you like to..."
❌ "The query returned..."
❌ "Based on the database results..."
❌ "According to the SQL execution..."

Remember: You're helping someone who wants to donate blood and save lives. Make it easy, quick, and encouraging!
"""

def analysis_agent(state: AgentState) -> AgentState:
    """Generate natural language answer from query results"""
    question = state["question"]
    sql_query = state["sql_query"]
    query_result = state["query_result"]
    
    prompt = f"""{ANALYSIS_AGENT_PROMPT}

Original Question: {question}

SQL Query Used: {sql_query}

Query Results:
{query_result}

Please generate a friendly and clear response based on the query results."""

    response = client.complete(
    messages=[
        SystemMessage(ANALYSIS_AGENT_PROMPT),
        UserMessage(prompt)
    ],
    model=model
    )

    final_answer = response.choices[0].message.content.strip()
    state["final_answer"] = final_answer
    
    return state

# Build the LangGraph workflow
def create_text2sql_graph():
    """Create the LangGraph state graph for Text2SQL with graph generation"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("duckdbsql_agent", duckdbsql_agent)
    workflow.add_node("executer_agent", executer_agent)
    workflow.add_node("analysis_agent", analysis_agent)

    
    # Add edges - start with guardrails check
    workflow.set_entry_point("duckdbsql_agent")
    workflow.add_edge("duckdbsql_agent", "executer_agent")
    workflow.add_edge("executer_agent", "analysis_agent")

    workflow.add_edge("analysis_agent", END)
    
    
    return workflow.compile()


# Create the compiled graph
text2sql_graph = create_text2sql_graph()

def run_text2sql_workflow(question: str) -> AgentState:
    """Run the Text2SQL workflow with the given question"""
    initial_state = AgentState(
        question=question,
        language="",
        schema="",
        sql_query="",
        query_result="",
        final_answer="",
        error="",
        iteration=0,
        needs_graph=False,
        graph_type="",
        graph_json="",
        is_in_scope=True
    )
    
    try:
        # Invoke the graph
        final_state = text2sql_graph.invoke(
            initial_state,
            config={"recursion_limit": 50}
        )
        
        return final_state
        
    except Exception as e:
        return {
            "error": str(e),
            "final_answer": f"An error occurred while processing your question: {str(e)}"
        }


def main():
    while True:
        user_question = input("Enter your question (or 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        
        result = run_text2sql_workflow(user_question)
        
        print("\n--- Assistant Response ---")
        print(result.get("final_answer", "No response generated."))
        print("\n--- SQL Query ---")
        print(result.get("sql_query", "No SQL query generated."))
        print("\n--- Query Result ---")
        print(result.get("query_result", "No query result."))

if __name__ == "__main__":
    main()