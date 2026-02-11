"""
Phase 2: Project
Goal: Cas complet de conversation avec mémoire et contexte
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

# Agent expert en RAG
rag_expert = Agent(
    role="Expert RAG",
    goal="Expliquer RAG et ses applications",
    backstory="Tu es un expert en RAG (Retrieval-Augmented Generation) avec 5 ans d'expérience en production",
    llm=llm
)

# Agent expert en agents
agent_expert = Agent(
    role="Expert Agents",
    goal="Expliquer les agents LLM et leur orchestration",
    backstory="Tu es un expert en agents autonomes et orchestration multi-agents",
    llm=llm
)

# Conversation collaborative
project_tasks = [
    Task(
        description="Explique pourquoi RAG + Agents est une combinaison puissante",
        agent=rag_expert,
        expected_output="Explication technique complète"
    ),
    Task(
        description="Quels sont les défis de combiner RAG avec des agents autonomes?",
        agent=agent_expert,
        expected_output="Les défis techniques et pratiques"
    ),
    Task(
        description="Donne un cas d'usage concret où tu combinerais RAG + Agents Orchestration",
        agent=rag_expert,
        expected_output="Un exemple pratique et détaillé"
    )
]

crew = Crew(agents=[rag_expert, agent_expert], tasks=project_tasks, verbose=False)
result = crew.kickoff()

print("=" * 70)
print("Phase 2: Project - RAG + Agents Collaboration")
print("=" * 70)
print("\n🤝 Conversation entre experts:\n")
print(result)
print("\n" + "=" * 70)
print("✨ Résultat: Multi-agent conversation avec mémoire de contexte")
print("=" * 70)
