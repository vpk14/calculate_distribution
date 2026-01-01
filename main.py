import streamlit as st
import math
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent
from dotenv import load_dotenv
import json
import re
from langchain.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

def parse_and_display_messages(messages_dict):
    """
    Parse LangChain messages and display them in Streamlit.
    
    Args:
        messages_dict: Dictionary with 'messages' key containing list of BaseMessage objects
    """
    st.title("Agent Conversation")
    
    messages = messages_dict.get('messages', [])
    
    for i, message in enumerate(messages):
        # HumanMessage - User input
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        
        # AIMessage - Agent thinking + tool calls + responses
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                # Show tool calls if present
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    st.subheader("🔧 Tool Calls")
                    for tool_call in message.tool_calls:
                        with st.expander(f"Tool: {tool_call['name']}"):
                            st.write(f"**Tool Name:** {tool_call['name']}")
                            st.write(f"**Tool ID:** {tool_call.get('id', 'N/A')}")
                            st.json(tool_call['args'], expanded=True)
                
                # Show AI response text
                if message.content:
                    st.subheader("Response")
                    st.write(message.content)
                
                # Show usage metadata
                if hasattr(message, 'usage_metadata') and message.usage_metadata:
                    with st.expander("📊 Token Usage"):
                        st.json(message.usage_metadata)
        
        # ToolMessage - Tool execution results
        elif isinstance(message, ToolMessage):
            with st.chat_message("tool"):
                st.info(f"**Tool Result:** {message.name}")
                st.write(message.content)
                if hasattr(message, 'tool_call_id'):
                    st.caption(f"Tool Call ID: {message.tool_call_id}")



@tool
def calculate_home_distribution(total_people: int, distance_km: float) -> str:
    """
    Calculates the distribution of people across homes given distance and population.
    Assumes 13.5 homes per km and ensures even numbers of people per home.
    """
    homes_per_km = 13.5
    total_homes = int(distance_km * homes_per_km)
    
    if total_homes == 0:
        return "Distance is too short to calculate homes."
    
    # Logic for even distribution
    avg = total_people / total_homes
    n = int(avg // 2) * 2  # Lower even number
    
    # Solve: n*x + (n+2)*y = total_people; x + y = total_homes
    y = int((total_people - (n * total_homes)) // 2)
    x = total_homes - y
    
    # Pattern Logic
    pattern_desc = ""
    if y > 0:
        ratio = round(x / y) if y != 0 else 0
        pattern_desc = f"Pattern: {ratio} homes with {n} people followed by 1 home with {n+2} people."
    else:
        pattern_desc = f"All homes receive exactly {n} people."

    return (f"Total Homes: {total_homes}\n"
            f"Result: {x} homes with {n} people and {y} homes with {n+2} people.\n"
            f"{pattern_desc}")

# --- 2. Setup Agent ---

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 
tools = [calculate_home_distribution]

agent = create_deep_agent(
    model=model,
    tools=tools,
    system_prompt=(
        "You are a logistics assistant for home distribution. "
        "Always use the calculate_home_distribution tool for distance/people queries. "
        "Explain the resulting pattern in simple terms / show the pattern for the user."
    )
)

# --- 3. Streamlit UI ---

st.set_page_config(page_title="Optimal People Distributor", layout="centered")

st.title("🏠 Home Distribution Agent")
st.markdown("Calculate how to distribute people across homes (13.5 homes/km) using even numbers.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        dist = st.number_input("Total Distance (km)", min_value=30.0, value=48.0, step = 1.0)
    with col2:
        ppl = st.number_input("Total People", min_value=2, value=6900, step=100)
    
    submit = st.form_submit_button("Calculate Distribution")

if submit:
    with st.spinner("Agent is calculating the optimal pattern..."):
        user_query = f"Calculate the distribution for {ppl} people across {dist} kms."
        
        # Invoke the agent
        response = agent.invoke({"messages": [{"role": "user", "content": user_query}]})
        
        # Display Results
        st.subheader("Agent's Strategy")
        parse_and_display_messages(response)
        print(response)

st.divider()
st.info("Note: This agent ensures every home has an even number of people by alternating between two adjacent even numbers.")