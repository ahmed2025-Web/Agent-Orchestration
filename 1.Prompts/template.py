from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv
import os

load_dotenv()

llm = LLM(
    model="mistral/mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

agent = Agent(
    role="Expliqueur",
    goal="Expliquer",
    backstory="Tu expliques bien",
    llm=llm
)

# TEMPLATE
template = "Explique {topic} en une phrase"

print("=" * 60)
print("TEMPLATE INTERACTIF")
print("=" * 60)
print(f"\nTemplate: {template}")
print("Entrez les topics à expliquer (séparés par des virgules)")
print("Exemple: LLM, RAG, Agents\n")

# input utilisateur
user_input = input("Topics: ")
topics = [t.strip() for t in user_input.split(",")]

print("\n" + "=" * 60)
print(f"Explication de: {topics}")
print("=" * 60)


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
