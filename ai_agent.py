import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import json
import pandas as pd
import duckdb
from datetime import datetime
import uuid
from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

class Message(TypedDict):
    """Message structure for conversation history"""
    role: str  # "user" or "assistant"
    content: str


def message_reducer(existing: list, new: list) -> list:
    """Reducer function to append new messages to existing messages"""
    if existing is None:
        existing = []
    if new is None:
        return existing
    return existing + new


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
    # LangGraph memory - messages accumulate automatically across invocations
    messages: Annotated[list[Message], message_reducer]


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

OPTIMIZED_SQL_SYSTEM_PROMPT = """You are a DuckDB SQL expert for blood donation events data.

## OUTPUT RULES
- Return ONLY the SQL query (no markdown, no backticks, no explanations)
- If unanswerable, return: NOT_ANSWERABLE

## TABLE & COLUMNS
Table: `blood_donation_events.csv` (NO quotes around table name)

Columns (all lowercase snake_case):
| Column | Type | Description |
|--------|------|-------------|
| event_day | TEXT | Day of week (Sunday, Monday, etc.) |
| event_date | DATE | Event date (YYYY-MM-DD format) |
| event_title | TEXT | Campaign name (UPPERCASE) |
| event_url | TEXT | Event webpage URL |
| organizer | TEXT | Hosting organization (UPPERCASE) |
| blood_donation_location | TEXT | Full venue address (UPPERCASE) |
| start_time | TEXT | Start time (e.g., "10.00 PAGI") |
| end_time | TEXT | End time (e.g., "5.00 PETANG") |
| blood_donor_target | INTEGER | Target donor count (0 = no target) |

## KEY RULES
1. **Table reference**: `FROM blood_donation_events.csv` (no quotes)
2. **Text matching**: Always use `ILIKE` for case-insensitive search
3. **Date filtering**: Use `event_date` with CURRENT_DATE for relative dates
4. **SELECT only**: No INSERT, UPDATE, DELETE, DROP, ALTER

## DATE PATTERNS
- Today: `WHERE event_date = CURRENT_DATE`
- This week: `WHERE event_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days'`
- This month: `WHERE YEAR(event_date) = YEAR(CURRENT_DATE) AND MONTH(event_date) = MONTH(CURRENT_DATE)`
- Specific date: `WHERE event_date = '2025-04-12'`

## QUERY EXAMPLES
Count events: `SELECT COUNT(*) AS total FROM blood_donation_events.csv`
Events in location: `SELECT * FROM blood_donation_events.csv WHERE blood_donation_location ILIKE '%bangi%' ORDER BY event_date`
By organizer: `SELECT * FROM blood_donation_events.csv WHERE organizer ILIKE '%kipmall%'`
Total targets: `SELECT SUM(blood_donor_target) AS total FROM blood_donation_events.csv`"""


# SQL Generation Agent
def format_messages_for_context(messages: list) -> str:
    """Format LangGraph messages for LLM context"""
    if not messages:
        return "No previous conversation."
    
    formatted = []
    # Keep last 10 messages (5 turns) for context
    recent_messages = messages[-10:] if len(messages) > 10 else messages
    
    for msg in recent_messages:
        role = "User" if msg.get("role") == "user" else "Assistant"
        formatted.append(f"{role}: {msg.get('content', '')}")
    
    return "\n".join(formatted)


def duckdbsql_agent(state: AgentState) -> AgentState:
    """Generate SQL query from natural language question"""
    question = state["question"]
    iteration = state.get("iteration", 0)
    messages = state.get("messages", [])
        
    # Get current date for context
    current_date = datetime.now()
    
    # Format conversation context from LangGraph memory
    history_context = format_messages_for_context(messages)
    
    prompt = f"""Generate a DuckDB SQL query for this question about blood donation events in Malaysia.

## CONTEXT
- Today: {current_date.strftime('%Y-%m-%d')} ({current_date.strftime('%A, %d %B %Y')})
- Previous conversation: {history_context}

## QUESTION
{question}

## DATASET INFO
- Table: `blood_donation_events.csv` (NO quotes)
- Each row = one blood donation event at one location on one date
- Same event_title can appear multiple times (different locations/dates)

## COLUMNS (all lowercase)
- event_day: Day of week (text, informational only)
- event_date: Event date (YYYY-MM-DD format, use for ALL date filtering)
- event_title: Campaign name (UPPERCASE text)
- organizer: Hosting organization (UPPERCASE text)
- blood_donation_location: Full venue address (UPPERCASE text, use ILIKE for search)
- start_time, end_time: Time strings (e.g., "10.00 PAGI", "5.00 PETANG")
- blood_donor_target: Integer (0 = no target specified)

## RULES
1. Return ONLY the SQL query - no markdown, no explanation
2. Use ILIKE for all text matching (case-insensitive)
3. Use event_date for date filtering (not event_day)
4. Table name without quotes: FROM blood_donation_events.csv
5. If question cannot be answered with this data, return: NOT_ANSWERABLE

## DATE EXAMPLES
- "today" → WHERE event_date = CURRENT_DATE
- "this week" → WHERE event_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days'
- "this month" → WHERE YEAR(event_date) = YEAR(CURRENT_DATE) AND MONTH(event_date) = MONTH(CURRENT_DATE)

Generate the SQL query now:"""

    
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b:free",
        messages=[
            {"role": "system", "content": OPTIMIZED_SQL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )

    raw_response = response.choices[0].message.content.strip()
    
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
    question = state.get("question", "")
    
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
            # Check if user is asking about a future date beyond available data
            # Get the max date in the dataset
            try:
                max_date_query = f"SELECT MAX(event_date) as max_date FROM '{csv_path}'"
                max_date_result = duckdb.sql(max_date_query).fetchdf()
                max_date = max_date_result['max_date'].iloc[0]
                
                # Extract date from user question if possible
                state["query_result"] = json.dumps({
                    "status": "no_results",
                    "max_available_date": str(max_date) if max_date else None,
                    "message": "No results found for this query."
                })
            except:
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

ANALYSIS_AGENT_PROMPT = """You are a friendly blood donation assistant for Malaysia. Transform database results into helpful, human-readable responses.

## RESPONSE STYLE
- **Concise**: 1-3 sentences for simple queries, structured lists for multiple events
- **Warm**: Friendly tone that encourages blood donation
- **Clear**: No technical jargon, no SQL references

## FORMAT BY RESPONSE TYPE

### Multiple Events (3+):
"I found [X] blood donation events! 🩸

📅 **[Day], [Date]**
   📍 [Full location]
   🕐 [Start Time] - [End Time]

 📅 **[Day], [Date]**
   📍 [Full location]
   🕐 [Start Time] - [End Time]"

### 1-2 Events:
"There's a blood donation event at **[Venue]** on **[Date]**! 🩸
📍 [Full Location]
🕐 [Start Time] - [End Time]"

### Counts/Statistics:
"There are **[number] blood donation events** [timeframe/location]. 🩸"

### No Results:
"I couldn't find any blood donation events matching your search. 😔"

### Future Date (Beyond Available Data):
When user asks about a date that's too far in the future (no data available yet):
"I don't have event information for [requested date] yet. 📅

Event schedules are typically updated closer to the date. Please check back about **1 week before** your requested date for the latest information! 🩸"

## DETECTING FUTURE DATE QUERIES
If the query result shows "no_results" or is empty, AND the user's question mentions a specific future date (like "January 2026", "next year", etc.), respond with the future date message above.

## FORMATTING RULES
✅ Convert times: "10.00 PAGI" → "10:00 AM", "5.00 PETANG" → "5:00 PM", "7.00 MALAM" → "7:00 PM"
✅ Convert dates: "2025-04-12" → "Saturday, 12th April 2025"
✅ Show FULL location as CLICKABLE GOOGLE MAPS LINK:
   - Format: [Full Address](https://www.google.com/maps/search/?api=1&query=URL_ENCODED_ADDRESS)
   - Example: [DEWAN SERBAGUNA TAMAN SRI WATAN, JALAN 6/3, 68000 AMPANG, SELANGOR](https://www.google.com/maps/search/?api=1&query=DEWAN+SERBAGUNA+TAMAN+SRI+WATAN%2C+JALAN+6%2F3%2C+68000+AMPANG%2C+SELANGOR)
   - URL encoding: Replace spaces with +, commas with %2C, slashes with %2F
   - ALWAYS make the full address a clickable Markdown link to Google Maps
✅ Use emojis sparingly: 🩸 📍 🕐 🎉 📅 (max 3-4 per response)
✅ Bold key info: dates, venue names, numbers
✅ Group events by date when listing multiple

## LANGUAGE
- Respond in the same language as the user's question
- English question → English response
- Malay question → Malay response
- Mixed → Prefer the dominant language

## DON'T
❌ Mention SQL, queries, or databases
❌ Make suggestions or recommendations (except for checking back later for future dates)
❌ Add unnecessary explanations
❌ Include technical details
❌ Use more than 4 emojis per response"""

def analysis_agent(state: AgentState) -> AgentState:
    """Generate natural language answer from query results"""
    question = state["question"]
    sql_query = state["sql_query"]
    query_result = state["query_result"]
    messages = state.get("messages", [])
    
    # Get current date for context
    current_date = datetime.now()
    
    # Format conversation context from LangGraph memory
    history_context = format_messages_for_context(messages)
    
    prompt = f"""Transform these blood donation event results into a friendly response.

**User's Question:** {question}

**Today's Date:** {current_date.strftime('%Y-%m-%d')} ({current_date.strftime('%A, %d %B %Y')})

**Query Results:**
{query_result}

**Previous Conversation:**
{history_context}

**IMPORTANT:** If the results are empty or show "no_results", check if the user is asking about a future date (e.g., dates in 2026 or beyond current available data). If so, politely explain that event information for that date isn't available yet and ask them to check back about 1 week before their requested date.

Generate a friendly, clear response following the formatting guidelines. Match the user's language (English/Malay)."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b:free",
        messages=[
            {"role": "system", "content": ANALYSIS_AGENT_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    final_answer = response.choices[0].message.content.strip()
    state["final_answer"] = final_answer

    # Add messages to LangGraph memory (user question + assistant response)
    # These will automatically accumulate via the message_reducer
    new_messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": final_answer}
    ]
    state["messages"] = new_messages
    
    return state

# Build the LangGraph workflow
def create_text2sql_graph():
    """Create the LangGraph state graph for Text2SQL with memory support"""
    
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
    
    # Add memory checkpointer for conversation persistence
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)


# Create the compiled graph with memory
text2sql_graph = create_text2sql_graph()

def run_text2sql_workflow(question: str, thread_id: str = None) -> AgentState:
    """Run the Text2SQL workflow with LangGraph memory
    
    Args:
        question: The user's question
        thread_id: Unique ID for the conversation thread (maintains history)
    
    Returns:
        AgentState with the final answer
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
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
        is_in_scope=True,
        messages=[]  # LangGraph memory handles accumulation via checkpointer
    )
    
    # Configuration with thread_id for memory persistence
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50
    }
    
    try:
        # Invoke the graph with memory-enabled config
        final_state = text2sql_graph.invoke(initial_state, config=config)
        return final_state
        
    except Exception as e:
        return {
            "error": str(e),
            "final_answer": f"An error occurred while processing your question: {str(e)}",
            "messages": []
        }


def main():
    """Main function with LangGraph memory-based conversation"""
    
    # Generate a unique thread ID for this conversation session
    # Same thread_id = same conversation memory
    thread_id = str(uuid.uuid4())
    
    print("\n🩸 Blood Donation Events Assistant")
    print("="*40)
    print("Ask me about blood donation events in Malaysia!")
    print("Type 'exit' to quit, 'new' to start a new conversation.\n")
    print(f"Session ID: {thread_id[:8]}...\n")
    
    while True:
        user_question = input("You: ").strip()
        
        if not user_question:
            continue
        
        if user_question.lower() == 'exit':
            print("\nThank you for using the Blood Donation Assistant. Goodbye! 👋")
            break
        
        if user_question.lower() == 'new':
            # Start a new conversation with fresh memory
            thread_id = str(uuid.uuid4())
            print(f"\n✅ New conversation started!")
            print(f"Session ID: {thread_id[:8]}...\n")
            continue
        
        # Run workflow with LangGraph memory (thread_id maintains conversation)
        result = run_text2sql_workflow(user_question, thread_id)
        
        final_answer = result.get("final_answer", "No response generated.")
        
        print(f"\nAssistant: {final_answer}")
        print()  # Empty line for readability
        
        # Debug: Show conversation history from memory
        # messages = result.get("messages", [])
        # print(f"[DEBUG] Messages in memory: {len(messages)}")


if __name__ == "__main__":
    main()