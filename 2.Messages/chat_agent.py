"""
Phase 2: Projet de Chat - Agent IA Simple
Objectif: Permettre à l'utilisateur de discuter en direct avec un Agent IA 
sans utiliser de documents externes, pour tester sa "personnalité" et ses connaissances générales.
"""

from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv
import os

# 1. Chargement des variables d'environnement
load_dotenv()

# 2. Configuration du LLM (Génération)
llm = LLM(
    model="mistral/mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

# 3. Création de l'Agent avec un Backstory détaillé
guide = Agent(
    role="Mentor en Intelligence Artificielle",
    goal="Accompagner l'utilisateur dans sa compréhension de l'IA et de l'orchestration d'agents.",
    backstory="""Tu es un mentor passionné et érudit nommé 'Orion'. 
    Tu as vu naître l'intelligence artificielle et tu en connais les moindres recoins, 
    de l'époque des systèmes experts jusqu'aux agents autonomes modernes.
    Ta mission est de rendre le savoir accessible. Tu es connu pour ton ton captivant, 
    tes analogies brillantes et ton enthousiasme communicatif. 
    Tu aimes ponctuer tes explications de conseils pour les futurs développeurs.
    Tu es très poli, tu encourages toujours l'utilisateur et tu t'adaptes à son niveau de connaissance.""",
    llm=llm,
    verbose=False
)

def start_simple_chat():
    print("=" * 60)
    print("🌟 CHAT INTERACTIF - MENTOR ORION")
    print("=" * 60)
    print("Posez vos questions sur l'IA ou l'orchestration d'agents.")
    print("Tapez 'quitter' ou 'exit' pour arrêter la conversation.\n")

    while True:
        # Demander la question à l'utilisateur
        user_input = input("👉 Votre question : ")

        if user_input.lower() in ['quitter', 'exit', 'quit']:
            print("\nÀ bientôt ! Que la curiosité vous guide toujours.")
            break

        if not user_input.strip():
            continue

        # Création de la tâche dynamique
        task = Task(
            description=f"Réponds de manière pédagogique à cette question de l'utilisateur : {user_input}",
            agent=guide,
            expected_output="Une réponse bienveillante, détaillée et structurée."
        )

        # Lancement de l'exécution
        crew = Crew(agents=[guide], tasks=[task], verbose=False)
        
        print("\n⏳ Orion réfléchit...")
        result = crew.kickoff()

        print("\n" + "✨" * 30)
        print(f"🤖 ORION :")
        print(result)
        print("✨" * 30 + "\n")

if __name__ == "__main__":
    try:
        start_simple_chat()
    except KeyboardInterrupt:
        print("\n\nInterruption détectée. Au revoir !")
