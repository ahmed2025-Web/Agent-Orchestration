"""
Templates de Prompts
Goal: Créer des templates réutilisables
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
# Utilisation des templates

agent = Agent(
    role="Expert",
    goal="Répondre avec expertise",
    backstory="Tu es un expert technique",
    llm=llm
)

template = "Explique {topic} en une phrase"


topics = ["LLM", "RAG", "Agents"]

print("=" * 60)
print("SIMPLE TEMPLATE")
print("=" * 60)
print(f"\nTemplate: {template}")
print(f"Topics: {topics}")
print("\n" + "=" * 60)


for topic in topics:
    prompt = template.format(topic=topic)
    task = Task(
        description=prompt,
        agent=agent,
        expected_output="Une phrase simple"
    )
    
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    result = crew.kickoff()
    
    print(f"\n {topic}:")
    print(f"{result}")

print("\n" + "=" * 60)