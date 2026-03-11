"""
Phase 1: Simple CrewAI avec Groq
"""

from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv
import os

load_dotenv()


llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Créer un agent simple
agent = Agent(
    role="Expert IA",
    goal="Expliquer les concepts d'IA",
    backstory="Tu es un expert en intelligence artificielle",
    llm=llm
)

# Créer une tâche simple
task = Task(
    description="Explique ce qu'est un LLM en une phrase",
    agent=agent,
    expected_output="Une phrase simple et claire"
)

# Exécuter
crew = Crew(agents=[agent], tasks=[task], verbose=True)
result = crew.kickoff()

print("\n" + "="*50)
print(result)
