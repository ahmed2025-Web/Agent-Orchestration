"""
Phase 3: Embeddings avec CrewAI
Goal: Comprendre comment CrewAI gère les embeddings.
"""

from crewai import Agent
from crewai.llm import LLM
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration de l'embedder pour CrewAI
# CrewAI utilise ces paramètres pour ses outils de recherche (SearchTools)
embedder_config = {
    "provider": "mistral",
    "config": {
        "model": "mistral-embed",
        "api_key": os.getenv("MISTRAL_API_KEY")
    }
}

print("=" * 60)
print("🧬 CONFIGURATION DES EMBEDDINGS (CrewAI style)")
print("=" * 60)
print(f"Provider: {embedder_config['provider']}")
print(f"Modèle: {embedder_config['config']['model']}")
print("=" * 60)
print("Note: Dans CrewAI, vous ne manipulez pas souvent les vecteurs directement.")
print("Vous passez cette configuration aux outils de recherche (SearchTools).")
print("=" * 60)
