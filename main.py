import os
from typing import TypedDict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun

# 1. SETUP: Define LLM and Search Tool
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)
search = DuckDuckGoSearchRun()

# 2. STATE
class AgentState(TypedDict):
    topic: str
    essay: str
    critique: Optional[str]
    revision_count: int

# 3. NODES

def researcher_node(state: AgentState):
    """The 'Student' now searches before writing."""
    print(f"\n--- RESEARCHER IS WORKING ON: {state['topic']} ---")
    topic = state['topic']
    critique = state.get('critique')
    count = state.get('revision_count', 0)

    # If it's a retry, we just rewrite. If it's new, we search first.
    if critique:
        print(f"   (Refining based on your feedback: '{critique}')")
        prompt = f"Rewrite this essay on '{topic}' to address this critique: {critique}. Keep it under 200 words."
    else:
        print("   (Searching the web...)")
        # Perform a real web search
        search_results = search.invoke(f"current facts about {topic}")
        print("   (Writing draft...)")
        prompt = f"Write a short, factual essay on '{topic}' using this search data: {search_results}. Keep it under 200 words."

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"essay": response.content, "revision_count": count + 1}

def human_node(state: AgentState):
    """The 'Teacher' is now YOU."""
    print("\n--- HUMAN REVIEW REQUIRED ---")
    print(f"\nCURRENT DRAFT:\n{state['essay']}\n")
    print("-" * 40)
    
    # This pauses the program and waits for your typing
    feedback = input("Type your critique (or press Enter to finish): ")
    
    return {"critique": feedback}

# 4. CONDITIONAL LOGIC
def should_continue(state: AgentState):
    """If you typed feedback, loop back. If empty, finish."""
    critique = state.get('critique')
    if critique and critique.strip() != "":
        return "rewrite"
    else:
        return "end"

# 5. BUILD THE GRAPH
builder = StateGraph(AgentState)

builder.add_node("researcher", researcher_node)
builder.add_node("human", human_node)

builder.set_entry_point("researcher")
builder.add_edge("researcher", "human")

builder.add_conditional_edges(
    "human",
    should_continue,
    {
        "rewrite": "researcher",
        "end": END
    }
)

graph = builder.compile()

# === VISUALIZATION (Feature #3) ===
try:
    print("\nGenerating graph image...")
    png_data = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
    print("Graph saved as 'graph.png'!")
except Exception as e:
    print(f"Could not draw graph (missing library?): {e}")

# 6. RUN IT
topic = input("\nWhat topic should I research? ")
initial_state = {"topic": topic, "revision_count": 0}
graph.invoke(initial_state)

print("\n=== FINAL OUTPUT ACCEPTED ===")
