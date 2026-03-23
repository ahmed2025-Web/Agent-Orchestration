"""
Phase 3: Retrieval avec CrewAI Tools
Goal: Utiliser un outil de recherche de manière isolée pour récupérer du contexte.
"""

from crewai_tools import TXTSearchTool
from dotenv import load_dotenv
import os

load_dotenv()

# Bypass validation OpenAI
os.environ["OPENAI_API_KEY"] = "sk-placeholder"

# 1. Initialiser l'outil (même configuration que vector_store.py)
tool = TXTSearchTool(
    txt='3.VectorDatabases/knowledge.txt',
    config={
        "embedder": {
            "provider": "mistral",
            "config": {
                "model": "mistral-embed",
                "api_key": os.getenv("MISTRAL_API_KEY")
            }
        }
    }
)

# 2. Simuler une recherche (retrieval)
query = "C'est quoi le Model Context Protocol ?"

print("=" * 60)
print("🔍 RETRIEVAL VIA CREWAI TOOL")
print("=" * 60)
print(f"Question: {query}\n")

# L'outil cherche dans le fichier texte et renvoie les passages pertinents
result = tool.run(search_query=query)

print(f"📄 Résultats trouvés par l'outil :\n")
print(result)
print("-" * 60)
print("\nNote: L'outil a converti la question en vecteur, cherché dans")
print("l'index local, et extrait les morceaux de texte correspondants.")
print("=" * 60)
