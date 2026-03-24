"""
Phase 3: RAG Pipeline complet avec CrewAI
Objectif: Combiner le "Retrieval" (recherche) et la "Generation" (synthèse par LLM) 
au sein d'un Agent autonome CrewAI.
"""

from crewai import Agent, Task, Crew
from crewai.llm import LLM
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
import os

load_dotenv()

# CrewAI tools require a dummy OPENAI_API_KEY to be set, even when not using OpenAI.
os.environ["OPENAI_API_KEY"] = "not_used"

# 1. Configuration du modèle de langage pour la Génération (Mistral)
llm = LLM(
    model="mistral/mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

# 2. Configuration de l'accès à la base vectorielle (Retrieval via Mistral Embeddings)
rag_tool = PDFSearchTool(
    pdf='3.VectorDatabases/metaciv.pdf',
    collection_name='metaciv_rag',
    config={
        "embedding_model": {
            "provider": "sentence-transformer",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            }
        }
    }
)

# 3. Création de l'Agent CrewAI (Il fera le lien entre LLM et l'Outil)
expert = Agent(
    role="Expert en Simulation Multi-Agents",
    goal="Fournir des explications précises de concepts en s'appuyant uniquement sur les documents fournis.",
    backstory="Tu es un chercheur méticuleux. Tu refuses d'inventer des informations et tu utilises toujours tes outils de recherche documentaires pour trouver tes réponses.",
    tools=[rag_tool],
    llm=llm,
    verbose=False # Affiche les coulisses ("pensées" de l'agent)
)

# 4. Création de la tâche qui va déclencher le pipeling RAG
task = Task(
    description="Explique en français la notion de 'cogniton' d'après le PDF fourni.",
    agent=expert,
    expected_output="Une réponse claire, résumée et en français, issue exclusivement du document PDF."
)

# 5. Lancement de l'orchestration
crew = Crew(agents=[expert], tasks=[task], verbose=False)

print("=" * 60)
print("🚀 DÉMARRAGE DU PIPELINE RAG (AGENT + VECTOR STORE)")
print("=" * 60)

result = crew.kickoff()

print("\n" + "=" * 60)
print("🤖 RÉPONSE FINALE GÉNÉRÉE :")
print("-" * 60)
print(result)
print("=" * 60)
