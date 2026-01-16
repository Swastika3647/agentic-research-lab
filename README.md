# 🧠 Agentic Research & Data Lab

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-green)
![OpenAI](https://img.shields.io/badge/AI-OpenAI%20GPT--4o-orange)

An autonomous AI Agent that doesn't just chat—it **thinks**, **researches**, and **codes**.

Built with **LangGraph**, this agent features a self-correcting routing system that intelligently switches between casual conversation, deep web research (Tavily), and data analysis (Python REPL).

---

## 🚀 Features

* **🧠 Smart Routing:** The agent automatically detects intent.
    * *Casual?* → Route to Chat.
    * *News/Info?* → Route to Web Researcher.
    * *Data/Math?* → Route to Python Analyst.
* **📉 Real-Time Visualization:** Generates and executes Python code to plot charts and graphs directly in the chat interface.
* **🛠️ Visual "Brain":** Powered by LangGraph to visualize the agent's decision-making process.
* **🚦 Safety Brake (Human-in-the-loop):** *Optional:* Configured to pause before executing code in LangGraph Studio, allowing human review.

## 🛠️ Tech Stack

* **Orchestration:** [LangGraph](https://langchain-ai.github.io/langgraph/)
* **Frontend:** [Streamlit](https://streamlit.io/)
* **LLM:** OpenAI GPT-4o-mini
* **Tools:**
    * **Tavily API:** For real-time web search results.
    * **Python REPL:** For executing code and simulations.
    * **Matplotlib:** For generating data visualizations.

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Swastika3647/agentic-research-lab.git](https://github.com/Swastika3647/agentic-research-lab.git)
cd agentic-research-lab
