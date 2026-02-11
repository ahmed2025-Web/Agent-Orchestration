"""
Phase 2: Messages Basic
Goal: Échange simple message-réponse
"""

from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv
import os

load_dotenv()

# Créer le LLM Mistral
llm = LLM(
    model="mistral/mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

# Agent
agent = Agent(
    role="Assistant",
    goal="Répondre simplement aux questions",
    backstory="Tu es un assistant utile et concis",
    llm=llm
)

# Une simple question
task = Task(
    description="Qu'est-ce qu'un embedding?",
    agent=agent,
    expected_output="Une explication simple et claire"
)

crew = Crew(agents=[agent], tasks=[task], verbose=False)
result = crew.kickoff()

print("=" * 60)
print("Phase 2: Messages Basic - Single Turn")
print("=" * 60)
print(f"Question: Qu'est-ce qu'un embedding?\n")
print(f"Réponse:\n{result}")
print("=" * 60)
