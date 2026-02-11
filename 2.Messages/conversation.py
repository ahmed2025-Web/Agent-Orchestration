"""
Phase 2: Conversation
Goal: Multi-tour conversation avec mémoire
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

# Agent avec mémoire
agent = Agent(
    role="Assistant Conversationnel",
    goal="Avoir une conversation naturelle en se souvenant du contexte",
    backstory="Tu es un assistant qui se souvient de la conversation précédente et y réponds logiquement",
    llm=llm
)

# Conversation multi-tours
conversation_tasks = [
    Task(
        description="Je viens de découvrir les agents LLM. Qu'est-ce que c'est?",
        agent=agent,
        expected_output="Explication initiale"
    ),
    Task(
        description="Comment ça fonctionne avec les tools?",
        agent=agent,
        expected_output="Explication sur les tools et leur intégration"
    ),
    Task(
        description="Et la mémoire? Comment un agent se souvient?",
        agent=agent,
        expected_output="Explication sur la gestion de la mémoire"
    ),
    Task(
        description="Résume ce qu'on vient de discuter",
        agent=agent,
        expected_output="Un résumé cohérent de la conversation"
    )
]

crew = Crew(agents=[agent], tasks=conversation_tasks, verbose=False)
result = crew.kickoff()

print("=" * 60)
print("Phase 2: Conversation - Multi-Turn with Memory")
print("=" * 60)
print("\n📝 Conversation multi-tours:\n")
print(result)
print("\n" + "=" * 60)
print("💡 Concept: La mémoire permet au LLM de comprendre le contexte")
print("=" * 60)
