"""
Phase 1: Static Prompt
Goal: Prompt fixe sans variables
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
    role="Assistant",
    goal="Expliquer les concepts",
    backstory="Tu es un assistant utile",
    llm=llm
)

# Tâche avec prompt fixe
task = Task(
    description="Explique ce qu'est un LLM en une phrase",
    agent=agent,
    expected_output="Une phrase simple"
)

# Exécuter
crew = Crew(agents=[agent], tasks=[task], verbose=False)
result = crew.kickoff()
print(result)
