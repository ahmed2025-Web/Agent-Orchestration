"""
Phase 2: Messages Chat
Goal: Conversation avec rôles définis
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
    role="Professeur d'IA",
    goal="Enseigner les concepts d'IA de manière pédagogique",
    backstory="Tu es un professeur expert qui explique les concepts complexes de manière simple et accessible aux étudiants",
    llm=llm
)

# Conversation guidée
tasks = [
    Task(
        description="Explique ce qu'est une vector database",
        agent=agent,
        expected_output="Une explication pédagogique"
    ),
    Task(
        description="Pourquoi c'est important pour RAG?",
        agent=agent,
        expected_output="Les raisons clairement expliquées"
    )
]

crew = Crew(agents=[agent], tasks=tasks, verbose=False)
result = crew.kickoff()

print("=" * 60)
print("Phase 2: Messages Chat - Role-Based Conversation")
print("=" * 60)
print(f"Agent: Professeur d'IA\n")
print(f"Conversation:\n{result}")
print("=" * 60)
