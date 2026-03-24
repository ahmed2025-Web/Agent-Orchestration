"""
Phase 3: Retrieval (Recherche sémantique)
Objectif: Interroger la base de données vectorielle (précédemment créée) 
pour récupérer les passages du document PDF les plus pertinents par rapport à une question.
"""

from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = "sk-placeholder"

# 1. Connexion à notre base vectorielle existante
tool = PDFSearchTool(
    pdf='3.VectorDatabases/metaciv.pdf',
    collection_name='metaciv_rag',
    config={
        "embedding_model": {
            "provider": "sentence-transformer",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        }
    }
)

# 2. La question de l'utilisateur
query = "Qu'est-ce qu'un cogniton dans le document ?"

print("=" * 60)
print("🔍 SIMULATION DU RETRIEVAL (RECHERCHE)")
print("=" * 60)
print(f"Question posée : '{query}'\n")

# 3. L'outil compare le vecteur de la question avec les vecteurs du PDF
# et nous retourne le texte d'origine.
result = tool.run(query=query)

print("📄 Contexte récupéré depuis la base vectorielle :")
print("-" * 60)
print(result)
print("-" * 60)
print("\nCe texte brut sera utilisé dans la prochaine étape comme contexte pour le LLM.")
print("=" * 60)
