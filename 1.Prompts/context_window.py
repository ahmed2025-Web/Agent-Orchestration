"""
Phase 1: Context Window
Goal: Comprendre la limite de contexte
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
    goal="Gérer le contexte",
    backstory="Tu gères le contexte avec soin",
    llm=llm
)

print("=" * 70)
print("CONTEXT WINDOW - Limite de tokens")
print("=" * 70)

models = [
    ("Mistral Small", 32768, 24576),
    ("Mistral Medium", 131072, 98304),
    ("Claude Opus", 200000, 150000),
    ("Gemini 2.5 Pro", 1000000, 750000),
]

print("\n Comparaison des modèles:\n")
print(f"{'Modèle':<20} {'Tokens Max':<15} {'Mots Approx':<15} {'Pages (±)':<10}")
print("-" * 60)

for model, tokens, words in models:
    pages = words // 250  # ~250 words per page
    print(f"{model:<20} {tokens:<15,} {words:<15,} {pages:<10}")

print("\n" + "=" * 70)
print(" IMPLICATIONS PRATIQUES:")
print("=" * 70)

task = Task(
    description="""Explique pourquoi la limite de contexte est importante:
    1. Comment elle impacte les conversations longues?
    2. Comment gérer les gros documents?
    3. Quel est le lien avec la mémoire et RAG?""",
    agent=agent,
    expected_output="Une explication claire et structurée"
)

crew = Crew(agents=[agent], tasks=[task], verbose=False)
result = crew.kickoff()

print(f"\n{result}")
print("\n" + "=" * 70)
