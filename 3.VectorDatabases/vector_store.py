"""
Phase 3: Vector Store (Création de la base de données vectorielle)
Objectif: Charger un PDF, le découper (chunking), le transformer en vecteurs (embeddings) 
et le stocker localement en utilisant les outils intégrés de CrewAI.
"""

from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
import os

load_dotenv()

# CrewAI/Embedchain nécessite parfois cette variable même avec d'autres fournisseurs
os.environ["OPENAI_API_KEY"] = "sk-placeholder"

# Configuration de l'outil avec Gemini pour les embeddings
print("=" * 60)
print("📦 CRÉATION DU VECTOR STORE (PDF) AVEC GEMINI")
print("=" * 60)
print("Initialisation de l'outil PDFSearchTool...")

# 1. Création et stockage de la base vectorielle automatique
# Lors de son initialisation, cet outil lit le PDF, crée les embeddings Gemini
# et sauvegarde la base de données vectorielle localement (dossier ./db).
vector_store_tool = PDFSearchTool(
    pdf='3.VectorDatabases/metaciv.pdf',
    collection_name='metaciv_rag',
    config={
        "embedder": {
            "provider": "google",
            "config": {
                "model": "models/text-embedding-004",
                "api_key": os.getenv("GOOGLE_API_KEY")
            }
        }
    }
)

print("✅ Base de données vectorielle créée avec succès !")
print(f"Fichier source : 3.VectorDatabases/metaciv.pdf")
print("Modèle d'embedding : models/text-embedding-004")
print("=" * 60)