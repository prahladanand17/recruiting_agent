from agent.tools import tool_schema
import streamlit as st
from agent.agent import Agent
from email_handler.email_indexer import EmailIndexer

import logging

email_indexer = EmailIndexer(debug=True, log_level=logging.DEBUG)
email_indexer.write_emails_to_chroma() #TODO: Remove this after testing

agent_prompt = """You are a recruiting email assistant helping a senior software engineer evaluate job opportunities.

The user is looking for:
- Agent engineering, AI backend, or applied AI roles
- NOT data engineering, DevOps, or frontend roles
- Series A, B, or C stage companies preferred
- NYC office or fully remote
- Base compensation above $170K
- NOT interested in roles with 25%+ travel

You have access to these tools:
- get_recruiting_emails: fetch emails by metadata (date range, sender, read status)
- search_emails: semantic search over email content
- research_company: look up company info via web search
- evaluate_fit: score a role against the user's criteria (MUST call research_company first)
- draft_response: generate an email reply (interested or decline)

Guidelines:
- Always research a company before evaluating fit
- When asked about interesting/good fit emails, evaluate ALL retrieved emails
- When drafting responses, include enough context to make the email feel personal
- Never send emails — only draft them for user review
- Be concise in your final response — lead with the answer, details second"""

recruiting_agent = Agent(name="recruiting_agent", prompt=agent_prompt, tools=tool_schema)   

st.set_page_config(page_title="Recruiting Agent", layout="centered")

st.title("🤖 Recruiting Email Agent")
st.caption("Ask me about your recruiting emails")

# example queries to help user get started
with st.expander("Example queries"):
    st.markdown("""
    - *How many recruiting emails have I gotten in the last week?*
    - *Are there any interesting emails from agent engineering companies?*
    - *Did I get any emails from Hebbia or Cursor?*
    - *Find emails about RAG roles*
    - *Draft a response to the Sierra AI email expressing interest*
    - *Decline the data engineering roles from this week*
    """)

# initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# render conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chat input
if query := st.chat_input("Ask about your recruiting emails..."):
    
    # add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # run agent and stream response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # build history for agent — exclude current message
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if isinstance(m["content"], str)
            ]
        
        response = st.write_stream(recruiting_agent.run(query, history))
    
    # save response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
