"""
Phase 3: Vector Store (CrewAI SearchTools)
Goal: Utiliser un outil CrewAI qui gère automatiquement le stockage vectoriel.
"""

from crewai_tools import TXTSearchTool
from dotenv import load_dotenv
import os

load_dotenv()

# Note: CrewAI tools may check for OPENAI_API_KEY even if using another provider.
# On définit une valeur fictive pour passer la validation Pydantic.
os.environ["OPENAI_API_KEY"] = "sk-placeholder"

# 1. Définir l'outil de recherche
# L'outil va automatiquement créer un index vectoriel à partir du fichier texte
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

print("=" * 60)
print("📦 VECTOR STORE VIA CREWAI TOOLS")
print("=" * 60)
print(f"Fichier indexé: 3.VectorDatabases/knowledge.txt")
print(f"Outil utilisé: TXTSearchTool")
print("=" * 60)
print("Note: Au premier lancement, l'outil crée un dossier local (souvent './db')")
print("qui contient la base de données vectorielle gérée par Embedchain.")
print("=" * 60)
