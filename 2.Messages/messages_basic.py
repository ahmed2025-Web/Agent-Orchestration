"""
Phase 2: Messages Basic
Goal: Échange simple single-turn (un message → une réponse)
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
    role="Assistant IA",
    goal="Répondre clairement à une question simple",
    backstory="Tu es un assistant qui répond de façon concise et précise",
    llm=llm
)


task = Task(
    description="Qu'est-ce qu'un embedding en IA ?",
    agent=agent,
    expected_output="Une explication claire en quelques phrases"
)


crew = Crew(agents=[agent], tasks=[task], verbose=False)
result = crew.kickoff()

print("=" * 60)
print(" MESSAGES BASIC - Single Turn")
print("=" * 60)
print(f"Question: Qu'est-ce qu'un embedding en IA ?\n")
print(f"Réponse: {result}")
print("=" * 60)
