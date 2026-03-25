"""
Phase 3: Projet Final - Chat Interactif RAG
Objectif: Permettre à l'utilisateur de discuter en direct avec l'expert IA 
en utilisant le contenu du PDF comme base de connaissances.
"""

from crewai import Agent, Task, Crew
from crewai.llm import LLM
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
import os

# 1. Chargement des variables d'environnement
load_dotenv()

# Bypass pour la vérification OpenAI du framework (même si on utilise Mistral/HuggingFace)
# On utilise la clé Mistral si elle existe, sinon un placeholder
os.environ["OPENAI_API_KEY"] = os.getenv("MISTRAL_API_KEY") or "not_used"

# 2. Configuration du LLM (Génération)
llm = LLM(
    model="mistral/mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

# 3. Configuration de l'Outil de Recherche (Retrieval local)
# On utilise la collection 'metaciv_rag' créée précédemment
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

# 4. Création de l'Agent
expert = Agent(
    role="Expert en Systèmes Multi-Agents",
    goal="Répondre aux questions de l'utilisateur en utilisant exclusivement le document fourni.",
    backstory="""Tu es le Dr. Aris, un chercheur de renommée mondiale qui a consacré sa carrière à l'étude du framework MetaCiv. 
    Tu es reconnu pour ta rigueur scientifique et ta capacité à expliquer des concepts complexes comme les 'cognitons' 
    ou les 'environnements multi-agents' de manière limpide.
    Ton éthique de travail est irréprochable : tu ne parles que de ce qui est prouvé et documenté. 
    Si une question sort du cadre du document PDF fourni, tu l'expliques avec pédagogie en précisant que tes 
    connaissances actuelles sont limitées aux archives MetaCiv dont tu as la charge.
    Tu t'exprimes toujours dans un français impeccable, avec une pointe de bienveillance et un grand professionnalisme.""",
    tools=[rag_tool],
    llm=llm,
    verbose=False
)

def start_chat():
    print("=" * 60)
    print("🤖 CHAT INTERACTIF RAG - EXPERT METACIV")
    print("=" * 60)
    print("Tapez 'quitter' ou 'exit' pour arrêter la conversation.\n")

    while True:
        # Demander la question à l'utilisateur
        user_input = input("👉 Votre question : ")

        if user_input.lower() in ['quitter', 'exit', 'quit']:
            print("\nAu revoir ! Merci pour cet échange.")
            break

        if not user_input.strip():
            continue

        # Création de la tâche dynamique basée sur l'input utilisateur
        task = Task(
            description=f"Réponds à la question suivante en te basant sur le PDF : {user_input}",
            agent=expert,
            expected_output="Une réponse claire et concise basée sur le document."
        )

        # Lancement de l'exécution
        crew = Crew(agents=[expert], tasks=[task], verbose=False)
        
        print("\n🔍 L'agent analyse le document...")
        result = crew.kickoff()

        print("\n" + "-" * 30)
        print(f"🤖 RÉPONSE :")
        print(result)
        print("-" * 30 + "\n")

if __name__ == "__main__":
    try:
        start_chat()
    except KeyboardInterrupt:
        print("\n\nInterruption détectée. Au revoir !")
