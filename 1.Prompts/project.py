"""
Phase 1: Project
Goal: Combiner les concepts de prompts dans un cas d'usage
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

# Agent expert
agent = Agent(
    role="Expert Recherche en IA",
    goal="Expliquer les concepts avancés d'IA",
    backstory="Tu es un chercheur expert en IA avec une spécialité en RAG et orchestration d'agents",
    llm=llm
)

# Tâche combinant les concepts
task = Task(
    description="Explique ce qu'est RAG (Retrieval-Augmented Generation) et pourquoi c'est important pour les LLMs",
    agent=agent,
    expected_output="Une explication technique complète et bien structurée"
)

# Exécuter
crew = Crew(agents=[agent], tasks=[task], verbose=False)
result = crew.kickoff()

print("=" * 60)
print("Phase 1 Project: RAG Explanation")
print("=" * 60)
print(result)
print("=" * 60)
