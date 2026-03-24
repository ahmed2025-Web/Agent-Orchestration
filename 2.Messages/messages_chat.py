"""
Goal: Interaction avec rôles explicites (System/User)
L'agent joue un rôle système spécifique qui influence ses réponses.
"""

from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv
import os

load_dotenv()


llm = LLM(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)


agent = Agent(
    role="Professeur d'Informatique",
    goal="Expliquer les concepts d'IA de manière pédagogique avec des analogies simples",
    backstory="""Tu es un professeur d'informatique spécialisé en Intelligence Artificielle.
    Tu expliques toujours avec des analogies du quotidien.
    Tu structures tes réponses avec des points numérotés.
    Tu termines toujours par un résumé en une phrase.""",
    llm=llm
)


task = Task(
    description="Explique ce qu'est une base de données vectorielle et pourquoi c'est utile pour l'IA",
    agent=agent,
    expected_output="Une explication pédagogique avec analogies, points numérotés et résumé"
)


crew = Crew(agents=[agent], tasks=[task], verbose=False)
result = crew.kickoff()

print("=" * 60)
print(" MESSAGES CHAT - Interaction avec Rôles")
print("=" * 60)
print(f" Rôle Système: Professeur d'Informatique")
print(f" Question User: Base de données vectorielle")
print("-" * 60)
print(f" Réponse:\n{result}")
print("=" * 60)
