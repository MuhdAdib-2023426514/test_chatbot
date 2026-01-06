import os
from datetime import datetime
import streamlit as st
from ai_agent import run_text2sql_workflow


def main():
    """Main Streamlit app function"""
    
    # Page configuration
    st.set_page_config(
        page_title="PDN Blood Donation Chatbot",
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            color: #DC143C;
            padding: 20px 0;
        }
        .chat-message {
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .user-message {
            background-color: #E3F2FD;
            border-left: 5px solid #2196F3;
        }
        .bot-message {
            background-color: #FFEBEE;
            border-left: 5px solid #DC143C;
        }
        .sql-query {
            background-color: #F5F5F5;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 12px;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>🩸 PDN Blood Donation Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Ask me about blood donation events in Malaysia!</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This chatbot helps you find blood donation events in Malaysia using AI.
        
        **Ask about:**
        - Events today/this week/this month
        - Events in specific locations
        - Event organizers and schedules
        - Donor targets and statistics
        """)
        
        st.divider()
        
        st.header("🔧 Options")
        show_sql = st.checkbox("Show SQL Query", value=False)
        show_results = st.checkbox("Show Raw Results", value=False)
        
        st.divider()
        
        st.header("📝 Example Questions")
        examples = [
            "How many events are happening today?",
            "Show me blood donation events in Bangi this week",
            "What events are organized by KIPMALL?",
            "Total donor target for KEMPEN DERMA DARAH",
            "Events happening this weekend",
            "Show all events in December 2025"
        ]
        
        for example in examples:
            if st.button(example, key=example):
                st.session_state.current_question = example
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! 👋 I'm your blood donation assistant. Ask me anything about blood donation events in Malaysia!"
        })
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display SQL query if available
            if "sql_query" in message and show_sql and message["sql_query"]:
                with st.expander("📊 SQL Query"):
                    st.code(message["sql_query"], language="sql")
            
            # Display raw results if available
            if "raw_results" in message and show_results and message["raw_results"]:
                with st.expander("📋 Raw Results"):
                    st.json(message["raw_results"])
    
    # Chat input
    if prompt := st.chat_input("Ask about blood donation events..."):
        st.session_state.current_question = prompt
    
    # Process question
    if "current_question" in st.session_state and st.session_state.current_question:
        question = st.session_state.current_question
        del st.session_state.current_question
        
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing your question..."):
                # Run the workflow
                result = run_text2sql_workflow(question)
                
                # Extract response
                response = result.get("final_answer", "I couldn't process your question.")
                sql_query = result.get("sql_query", "")
                query_result = result.get("query_result", "")
                error = result.get("error", "")
                
                # Display error if any
                if error:
                    st.error(f"❌ Error: {error}")
                
                # Display response
                st.markdown(response)
                
                # Show SQL query if enabled
                if show_sql and sql_query:
                    with st.expander("📊 SQL Query"):
                        st.code(sql_query, language="sql")
                
                # Show raw results if enabled
                if show_results and query_result:
                    with st.expander("📋 Raw Results"):
                        try:
                            results_data = json.loads(query_result) if isinstance(query_result, str) else query_result
                            st.json(results_data)
                        except:
                            st.text(query_result)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sql_query": sql_query,
                    "raw_results": query_result
                })


if __name__ == "__main__":
    main()
