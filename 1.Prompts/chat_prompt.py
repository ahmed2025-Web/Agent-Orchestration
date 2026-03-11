"""
Phase 1: Chat Prompt
Goal: Conversation avec rôles 
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

# Agent avec rôle spécifique 
agent = Agent(
    role="Assistant Pédagogue",
    goal="Aider à comprendre les concepts d'IA",
    backstory="Tu es un assistant pédagogique expert en IA qui explique les concepts de manière claire et accessible",
    llm=llm
)

# Tâche simulating conversation
task = Task(
    description="Explique ce qu'est le prompt engineering de manière pédagogique",
    agent=agent,
    expected_output="Une explication claire et pédagogique"
)

# Exécuter
crew = Crew(agents=[agent], tasks=[task], verbose=False)
result = crew.kickoff()
print(f"Agent Role: Assistant Pédagogue")
print(f"Topic: Prompt Engineering")
print(f"Result: {result}")
