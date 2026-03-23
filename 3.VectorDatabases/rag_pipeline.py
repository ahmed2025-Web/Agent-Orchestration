"""
Phase 3: RAG Pipeline avec CrewAI
Goal: Un Agent qui utilise un outil de recherche pour répondre à une question.
C'est la forme la plus courante de RAG dans CrewAI.
"""

from crewai import Agent, Task, Crew
from crewai.llm import LLM
from crewai_tools import TXTSearchTool
from dotenv import load_dotenv
import os

load_dotenv()

# Bypass validation OpenAI
os.environ["OPENAI_API_KEY"] = "sk-placeholder"

# 1. Configurer le LLM Mistral
llm = LLM(
    model="mistral/mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

# 2. Définir l'outil RAG
rag_tool = TXTSearchTool(
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

# 3. Créer l'Agent Expert avec l'outil
expert = Agent(
    role="Expert en Architecture IA",
    goal="Répondre aux questions techniques en utilisant EXCLUSIVEMENT les documents fournis",
    backstory="Tu es un expert qui base toutes ses réponses sur des sources documentaires précises.",
    tools=[rag_tool],
    llm=llm,
    verbose=True # Pour voir l'agent réfléchir et utiliser l'outil
)

# 4. Définir la tâche
task = Task(
    description="Explique ce qu'est le MCP (Model Context Protocol) et donne son analogie.",
    agent=expert,
    expected_output="Une explication précise basée sur le document fourni."
)

# 5. Exécuter le Crew
crew = Crew(agents=[expert], tasks=[task])

print("=" * 60)
print("🚀 RAG PIPELINE VIA CREWAI AGENT")
print("=" * 60)

result = crew.kickoff()

print("\n" + "=" * 60)
print("🤖 RÉPONSE FINALE DE L'AGENT :")
print("-" * 60)
print(result)
print("=" * 60)
