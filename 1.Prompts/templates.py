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
    model="mistral/mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

#  TEMPLATES

# Template 1
template_explain = "Explique {topic} en une phrase simple"

# Template 2
template_technical = "Explique techniquement comment {concept} fonctionne et donne 2 exemples"

# Template 3
template_usecase = "Donne 3 cas d'usage réalistes pour {technology}"

# Template 4
template_compare = "Compare {option1} et {option2} en termes de performance et coût"

# Utilisation des templates

agent = Agent(
    role="Expert",
    goal="Répondre avec expertise",
    backstory="Tu es un expert technique",
    llm=llm
)

print("=" * 70)
print("TEMPLATES DE PROMPTS - Exemples d'utilisation")
print("=" * 70)

# Utilisation du template 1
tasks_template1 = [
    Task(
        description=template_explain.format(topic="LLM"),
        agent=agent,
        expected_output="Explication simple"
    ),
    Task(
        description=template_explain.format(topic="Vector Database"),
        agent=agent,
        expected_output="Explication simple"
    ),
]

# Utilisation du template 2
tasks_template2 = [
    Task(
        description=template_technical.format(concept="Embeddings"),
        agent=agent,
        expected_output="Explication technique"
    ),
]

# Utilisation du template 3
tasks_template3 = [
    Task(
        description=template_usecase.format(technology="RAG"),
        agent=agent,
        expected_output="Cas d'usage"
    ),
]

# Utilisation du template 4
tasks_template4 = [
    Task(
        description=template_compare.format(option1="Mistral", option2="Claude"),
        agent=agent,
        expected_output="Comparaison"
    ),
]

print("\n Template 1: Explication simple")
print(f"Format: {template_explain}")
print(f"Utilisation 1: {template_explain.format(topic='LLM')}")
print(f"Utilisation 2: {template_explain.format(topic='Vector Database')}")

crew1 = Crew(agents=[agent], tasks=tasks_template1, verbose=False)
result1 = crew1.kickoff()
print(f"\nRésultat:\n{result1}\n")

print("=" * 70)
print("\n Template 2: Question technique")
print(f"Format: {template_technical}")
print(f"Utilisation: {template_technical.format(concept='Embeddings')}")

crew2 = Crew(agents=[agent], tasks=tasks_template2, verbose=False)
result2 = crew2.kickoff()
print(f"\nRésultat:\n{result2}\n")

print("=" * 70)
print("\n Template 3: Cas d'usage")
print(f"Format: {template_usecase}")
print(f"Utilisation: {template_usecase.format(technology='RAG')}")

crew3 = Crew(agents=[agent], tasks=tasks_template3, verbose=False)
result3 = crew3.kickoff()
print(f"\nRésultat:\n{result3}\n")

print("=" * 70)
print("\n Template 4: Comparaison")
print(f"Format: {template_compare}")
print(f"Utilisation: {template_compare.format(option1='Mistral', option2='Claude')}")

crew4 = Crew(agents=[agent], tasks=tasks_template4, verbose=False)
result4 = crew4.kickoff()
print(f"\nRésultat:\n{result4}\n")

