"""
Phase 1: Tokens
Goal: Comprendre et compter les tokens
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
    role="Analyseur",
    goal="Analyser les tokens",
    backstory="Tu analyses le coût en tokens",
    llm=llm
)

# Exemples de prompts de différentes longueurs
prompts = [
    ("Court", "Qu'est-ce qu'un LLM ?"),
    ("Moyen", "Explique les agents LLM et leur rôle dans l'orchestration. Donne 3 exemples."),
    ("Long", "Explique en détail les agents LLM, leur architecture, comment ils fonctionnent avec les outils, les patterns de réponse (ReAct), la gestion de la mémoire, et donne au moins 5 cas d'usage réels avec RAG, multi-agents, et orchestration.")
]

print("=" * 70)
print("ESTIMATION DES TOKENS PAR LONGUEUR DE PROMPT")
print("=" * 70)

for name, prompt in prompts:
    # Estimation simple : ~1 token par mot, ~4 tokens par phrase
    words = len(prompt.split())
    estimated_tokens = max(words // 4, len(prompt.split()))
    
    task = Task(
        description=prompt,
        agent=agent,
        expected_output="Une réponse concise"
    )
    
    print(f"\n📝 Prompt {name}:")
    print(f"   Texte: {prompt[:60]}...")
    print(f"   Mots: ~{words}")
    print(f"   Tokens estimés: ~{estimated_tokens}")
    print(f"   Coût (Mistral): ~${estimated_tokens * 0.00001:.4f}")

print("\n" + "=" * 70)
print("Note: 1 token ≈ 4 caractères ou 0.75 mots")
print("Mistral Small: $0.14 per 1M input tokens")
print("=" * 70)
