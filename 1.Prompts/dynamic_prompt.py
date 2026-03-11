"""
Phase 1: Dynamic Prompt
Goal: Prompt avec variables
"""

from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv
import os

load_dotenv()

# Créer le LLM Groq
llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Agent
agent = Agent(
    role="Expert Technique",
    goal="Expliquer des concepts techniques",
    backstory="Tu es un expert en technologie",
    llm=llm
)

# Variable
topic = "vector databases"

# Tâche avec prompt dynamique
task = Task(
    description=f"Explique {topic} en une phrase simple",
    agent=agent,
    expected_output="Une explication claire et concise"
)

# Exécuter
crew = Crew(agents=[agent], tasks=[task], verbose=False)
result = crew.kickoff()
print(f"Topic: {topic}")
print(f"Result: {result}")
