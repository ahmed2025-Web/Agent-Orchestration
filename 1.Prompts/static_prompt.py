"""
Phase 1: Static Prompt
Goal: Prompt fixe sans variables
"""

from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv
import os

load_dotenv()

# Créer le LLM Mistral
llm = LLM(
    model="mistral/mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
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
