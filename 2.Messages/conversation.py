"""
Goal: Conversation entre 2 agents qui collaborent sur un sujet.
"""

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from dotenv import load_dotenv
import os

load_dotenv()


llm = LLM(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Agent 1 : Curieux qui pose des questions
curieux = Agent(
    role="Étudiant Curieux",
    goal="Poser des questions pertinentes sur l'IA et comprendre les réponses",
    backstory="""Tu es un étudiant en informatique qui découvre l'IA.
    Tu poses des questions claires et précises.
    Tu reformules ce que tu comprends pour vérifier.""",
    llm=llm
)

# Agent 2 : Expert qui répond
expert = Agent(
    role="Expert en IA",
    goal="Expliquer les concepts d'IA de manière accessible",
    backstory="""Tu es un chercheur en IA avec 10 ans d'expérience.
    Tu expliques simplement les concepts complexes.
    Tu donnes toujours des exemples concrets.""",
    llm=llm
)

# Tour 1 : L'étudiant pose une question initiale
task1 = Task(
    description="Pose une question sur ce qu'est RAG (Retrieval-Augmented Generation) et pourquoi c'est important",
    agent=curieux,
    expected_output="Une question claire et précise sur RAG"
)

# Tour 2 : L'expert répond à la question
task2 = Task(
    description="Réponds à la question de l'étudiant sur RAG de manière claire avec un exemple concret",
    agent=expert,
    expected_output="Une explication claire de RAG avec un exemple",
    context=[task1]  # L'expert voit la question du curieux
)

# Tour 3 : L'étudiant pose une question de suivi
task3 = Task(
    description="Basé sur l'explication reçue, pose une question de suivi sur les limites ou défis de RAG",
    agent=curieux,
    expected_output="Une question de suivi pertinente",
    context=[task2]  # Le curieux voit la réponse de l'expert
)


crew = Crew(
    agents=[curieux, expert],
    tasks=[task1, task2, task3],
    process=Process.sequential,
    verbose=False
)

result = crew.kickoff()

print("=" * 60)
print(" CONVERSATION MULTI-TURNS")
print("=" * 60)

print("\n Tour 1 - Étudiant pose une question:")
print(f"   {task1.output.raw if task1.output else 'N/A'}")

print("\n Tour 2 - Expert répond:")
print(f"   {task2.output.raw if task2.output else 'N/A'}")

print("\n Tour 3 - Étudiant pose une question de suivi:")
print(f"   {task3.output.raw if task3.output else 'N/A'}")


