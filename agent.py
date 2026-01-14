import os
from typing import TypedDict, List, Union
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load env keys automatically
load_dotenv()

# DEFINE STATE
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, dict]] # Added dict support
    next_step: str

# DEFINE TOOLS
search = TavilySearchResults(max_results=2)
python_repl = PythonREPL()

# --- HELPER FUNCTION ---
def get_message_text(message):
    """
    Helper to safely extract text from either a Message Object or a Dictionary.
    """
    if isinstance(message, dict):
        return message.get("content", "")
    return message.content

# DEFINE NODES

def router_node(state: AgentState):
    messages = state["messages"]
    # FIX: Use helper to safely get text
    last_message = get_message_text(messages[-1]).lower()
    
    if any(word in last_message for word in ["plot", "graph", "chart", "code", "execute", "python", "simulation"]):
        return {"next_step": "analyst"}
    elif any(word in last_message for word in ["search", "find", "news", "latest", "info on", "who is", "what is", "current"]):
        return {"next_step": "researcher"}
    else:
        return {"next_step": "chat"}

def chat_node(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def researcher_node(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = state["messages"]
    # FIX: Use helper
    query = get_message_text(messages[-1])
    
    try:
        search_result = search.invoke(query)
    except Exception as e:
        search_result = f"Error: {e}"
    
    prompt = f"You are a researcher. Answer using this data: {search_result}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

def analyst_node(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = state["messages"]
    # FIX: Use helper
    query = get_message_text(messages[-1])
    
    prompt = f"Write PYTHON CODE for: {query}. ONLY return code."
    code_response = llm.invoke([HumanMessage(content=prompt)])
    code = code_response.content.replace("```python", "").replace("```", "").strip()
    try:
        result = python_repl.run(code)
        output = f"Code executed:\n```python\n{code}\n```\n\nResult:\n{result}"
    except Exception as e:
        output = f"Error: {e}"
    return {"messages": [AIMessage(content=output)]}

# BUILD GRAPH
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("chat", chat_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    lambda x: x["next_step"],
    {"chat": "chat", "researcher": "researcher", "analyst": "analyst"}
)
workflow.add_edge("chat", END)
workflow.add_edge("researcher", END)
workflow.add_edge("analyst", END)

# COMPILE (No Checkpointer needed for Studio)
# OLD:
# graph = workflow.compile()

# NEW: Add 'interrupt_before'
graph = workflow.compile(interrupt_before=["analyst"])