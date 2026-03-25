"""
🚀 PROJET FINAL : AI STUDY PARTNER (L'Assistant d'Étude Intelligent)
Objectif : Transformer un document complexe (PDF) en un guide d'étude 
interactif avec résumé et quiz automatique.

Ce script démontre :
1. Recherche documentaire (RAG)
2. Coopération entre 3 agents (Multi-agents)
3. Transformation de savoir complexe en contenu pédagogique
"""

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
import os

# 1. Configuration initiale
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("MISTRAL_API_KEY") or "not_used"

llm = LLM(
    model="mistral/mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

# 2. Outil de recherche dans le PDF (Base de connaissance MetaCiv)
pdf_tool = PDFSearchTool(
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

# 3. Définition des Agents (L'Équipe Pédagogique)
archivist = Agent(
    role="Archiviste Scientifique",
    goal="Extraire les informations précises et techniques du document PDF.",
    backstory="""Tu es un expert en recherche documentaire. Ton rôle est de fouiller 
    dans les archives MetaCiv pour trouver la 'vérité brute' sur un sujet donné. 
    Tu ne simplifies rien, tu rapportes les faits tels qu'ils sont écrit.""",
    tools=[pdf_tool],
    llm=llm,
    verbose=True
)

pedagogue = Agent(
    role="Professeur de Vulgarisation",
    goal="Simplifier les concepts complexes pour les rendre accessibles à tous.",
    backstory="""Tu es un enseignant exceptionnel. Tu prends les notes techniques 
    de l'archiviste et tu les transformes en une explication claire, imagée 
    et facile à retenir. Tu utilises des analogies si nécessaire.""",
    llm=llm,
    verbose=True
)

quiz_master = Agent(
    role="Concepteur de Quiz",
    goal="Créer un test d'évaluation pour vérifier la compréhension de l'élève.",
    backstory="""Tu es expert en évaluation pédagogique. Tu crées des questions 
    pertinentes qui ciblent les points essentiels du sujet pour s'assurer 
    que l'étudiant a bien compris l'explication du professeur.""",
    llm=llm,
    verbose=True
)

def create_study_guide(topic):
    # 4. Définition des Tâches (Le flux de travail)
    search_task = Task(
        description=f"Cherche tout ce que le PDF dit sur le sujet suivant : {topic}",
        agent=archivist,
        expected_output="Un rapport technique détaillé contenant les citations du PDF."
    )

    summary_task = Task(
        description=f"Basé sur les recherches techniques, explique simplement le concept de {topic}.",
        agent=pedagogue,
        context=[search_task],
        expected_output="Une explication pédagogique en français, structurée et claire."
    )

    output_filename = f"3.VectorDatabases/fiche_{topic.replace(' ', '_')}.md"

    quiz_task = Task(
        description=f"""Crée le guide d'étude complet pour {topic}. 
        Le document doit contenir :
        1. L'explication pédagogique claire.
        2. Un quiz de 3 questions (avec réponses à la fin).
        """,
        agent=quiz_master,
        context=[summary_task],
        expected_output="Un document Markdown structuré avec titres et paragraphes.",
        output_file=output_filename # C'est ici qu'on crée le fichier !
    )

    # 5. Création et lancement du Crew
    study_crew = Crew(
        agents=[archivist, pedagogue, quiz_master],
        tasks=[search_task, summary_task, quiz_task],
        process=Process.sequential
    )

    print(f"\n🎓 Génération du guide d'étude pour : {topic}...")
    result = study_crew.kickoff()
    print(f"\n✅ Votre fiche de révision a été sauvegardée dans : {output_filename}")
    return result

if __name__ == "__main__":
    print("=" * 60)
    print("🌟 BIENVENUE DANS VOTRE AI STUDY PARTNER")
    print("=" * 60)
    
    sujet = input("👉 Quel concept du PDF voulez-vous étudier aujourd'hui ? ")
    
    if sujet.strip():
        resultat = create_study_guide(sujet)
        
        print("\n" + "📚" * 20)
        print("VOTRE GUIDE D'ÉTUDE PERSONNALISÉ")
        print("📚" * 20)
        print(resultat)
        print("=" * 60)
