import streamlit as st
import os
import matplotlib.pyplot as plt
from typing import TypedDict, List, Union
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import nest_asyncio
from dotenv import load_dotenv  # <--- NEW IMPORT

# FIX: Apply the event loop patch immediately for Streamlit
nest_asyncio.apply()

# LOAD SECRETS
load_dotenv()  # <--- READS YOUR .ENV FILE

# 1. SETUP UI CONFIG
st.set_page_config(page_title="Agentic Research & Data Lab", layout="wide")
st.title("🤖 Agentic Research & Data Lab")

# Sidebar for API Keys (Updated Logic)
st.sidebar.header("Configuration")

# Try to get keys from Environment (.env)
api_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

# If NOT found in .env, ask for them in the Sidebar
if not api_key:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
else:
    st.sidebar.success("✅ OpenAI Key Loaded")

if not tavily_key:
    tavily_key = st.sidebar.text_input("Tavily API Key", type="password")
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
else:
    st.sidebar.success("✅ Tavily Key Loaded")



# 2. DEFINE TOOLS
# Initialize tools only if keys are present
if tavily_key:
    search = TavilySearchResults(max_results=2)
else:
    search = None 

python_repl = PythonREPL()

# 3. DEFINE STATE
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    next_step: str

# 4. DEFINE NODES

def router_node(state: AgentState):
    """
    Decides where the query should go:
    1. Analyst: If asking for code, charts, or math.
    2. Researcher: If asking for external info, news, or specific facts.
    3. Chat: Default mode for writing, summarizing, or casual talk.
    """
    messages = state["messages"]
    last_message = messages[-1].content.lower()
    
    # Keyword detection (Simple Logic)
    if any(word in last_message for word in ["plot", "graph", "chart", "code", "execute", "python", "simulation"]):
        return {"next_step": "analyst"}
    elif any(word in last_message for word in ["search", "find", "news", "latest", "info on", "who is", "what is", "current"]):
        return {"next_step": "researcher"}
    else:
        # If it's just "write an essay", "hello", or general questions -> Chat
        return {"next_step": "chat"}

def chat_node(state: AgentState):
    """Normal LLM mode (no tools) - NEW NODE."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def researcher_node(state: AgentState):
    """Searches the web using Tavily."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = state["messages"]
    query = messages[-1].content
    
    if not search:
        return {"messages": [AIMessage(content="Error: Please enter a Tavily API Key in the sidebar.")]}
        
    try:
        # Search the web
        search_result = search.invoke(query)
    except Exception as e:
        return {"messages": [AIMessage(content=f"Search Error: {e}")]}
    
    # Synthesize the answer
    prompt = f"""
    You are a researcher. Answer the user's question using this data: 
    {search_result}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

def analyst_node(state: AgentState):
    """Executes Python code for data analysis."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = state["messages"]
    query = messages[-1].content
    
    prompt = f"""
    You are a data analyst. The user wants: "{query}".
    Write PYTHON CODE to solve this. 
    If they want a plot, use matplotlib.pyplot as plt.
    ONLY return the executable code, no markdown backticks.
    """
    code_response = llm.invoke([HumanMessage(content=prompt)])
    code = code_response.content.replace("```python", "").replace("```", "").strip()
    
    try:
        result = python_repl.run(code)
        output = f"I executed this code:\n```python\n{code}\n```\n\n**Result:**\n{result}"
        return {"messages": [AIMessage(content=output)]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error executing code: {e}")]}

# 5. BUILD GRAPH
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("router", router_node)
workflow.add_node("chat", chat_node)          # <--- Registered New Node
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)

# Set Entry Point
workflow.set_entry_point("router")

# Add Conditional Edges
workflow.add_conditional_edges(
    "router",
    lambda x: x["next_step"],
    {
        "chat": "chat",
        "researcher": "researcher", 
        "analyst": "analyst"
    }
)

# End Edges
workflow.add_edge("chat", END)
workflow.add_edge("researcher", END)
workflow.add_edge("analyst", END)

# Compile with Memory
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# 6. UI LOGIC
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Input Box
if prompt := st.chat_input("Ask me to research, plot a graph, or just chat..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    if api_key and tavily_key:
        config = {"configurable": {"thread_id": "1"}}
        inputs = {"messages": st.session_state.messages}
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                final_state = app.invoke(inputs, config=config)
                response = final_state["messages"][-1]
                st.write(response.content)
                
                # If a plot was generated, show it
                if "analyst" in final_state.get("next_step", "") or "plot" in prompt.lower() or "simulation" in prompt.lower():
                    st.pyplot(plt)
                    plt.clf()

        st.session_state.messages.append(response)
    else:
        st.error("Please enter BOTH API Keys in the sidebar!")