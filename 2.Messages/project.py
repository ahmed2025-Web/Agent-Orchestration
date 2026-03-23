"""
Goal: Mini-projet combinant tous les concepts de Messages.
Scénario : Un Interviewer pose 3 questions à un Expert, puis un Résumeur
synthétise l'échange complet.
"""

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from dotenv import load_dotenv
import os

load_dotenv()

# Créer le LLM Mistral
llm = LLM(
    model="mistral/mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

# ── Agents ──────────────────────────────────────────────────

interviewer = Agent(
    role="Journaliste Tech",
    goal="Poser des questions pertinentes et progressives sur l'orchestration d'agents IA",
    backstory="""Tu es un journaliste technologique pour un magazine spécialisé.
    Tu poses des questions claires qui vont du simple au complexe.
    Tu t'adresses à un public technique mais pas forcément expert.""",
    llm=llm
)

expert = Agent(
    role="Architecte IA",
    goal="Expliquer l'orchestration d'agents de manière technique mais accessible",
    backstory="""Tu es un architecte IA senior qui conçoit des systèmes multi-agents.
    Tu donnes des réponses structurées avec des exemples concrets.
    Tu utilises des analogies quand c'est utile.""",
    llm=llm
)

resumeur = Agent(
    role="Rédacteur en Chef",
    goal="Synthétiser une interview en un résumé structuré et engageant",
    backstory="""Tu es rédacteur en chef d'un magazine tech.
    Tu sais extraire les points clés d'une conversation.
    Tu produis des résumés clairs avec les points essentiels.""",
    llm=llm
)

# ── Tâches (Interview en 3 tours + résumé) ─────────────────

# Q1 : Question d'introduction
q1 = Task(
    description="Pose une question d'introduction sur ce qu'est l'orchestration d'agents IA et pourquoi c'est important aujourd'hui",
    agent=interviewer,
    expected_output="Une question d'introduction claire et engageante"
)

# R1 : Réponse à la question d'introduction
r1 = Task(
    description="Réponds à la question du journaliste sur l'orchestration d'agents IA",
    agent=expert,
    expected_output="Une réponse structurée avec définition et importance",
    context=[q1]
)

# Q2 : Question sur les patterns
q2 = Task(
    description="Basé sur la réponse, pose une question sur les différents patterns d'orchestration (séquentiel, parallèle, conditionnel)",
    agent=interviewer,
    expected_output="Une question technique sur les patterns",
    context=[r1]
)

# R2 : Réponse sur les patterns
r2 = Task(
    description="Explique les patterns d'orchestration avec des exemples concrets pour chacun",
    agent=expert,
    expected_output="Explication des patterns avec exemples",
    context=[q2]
)

# Q3 : Question sur le futur
q3 = Task(
    description="Pose une question finale sur les défis et le futur de l'orchestration d'agents",
    agent=interviewer,
    expected_output="Une question prospective",
    context=[r2]
)

# R3 : Réponse sur le futur
r3 = Task(
    description="Donne ta vision sur les défis actuels et l'avenir de l'orchestration d'agents",
    agent=expert,
    expected_output="Vision prospective avec défis et opportunités",
    context=[q3]
)

# Résumé final
resume = Task(
    description="""Synthétise l'interview complète en un résumé structuré avec :
    - Titre accrocheur
    - 3 points clés de l'interview
    - Citation marquante de l'expert
    - Conclusion en une phrase""",
    agent=resumeur,
    expected_output="Un résumé structuré de l'interview",
    context=[q1, r1, q2, r2, q3, r3]
)

# ── Exécution ───────────────────────────────────────────────

crew = Crew(
    agents=[interviewer, expert, resumeur],
    tasks=[q1, r1, q2, r2, q3, r3, resume],
    process=Process.sequential,
    verbose=False
)

result = crew.kickoff()

print("=" * 60)
print("🎙️  PROJET: INTERVIEW IA - Orchestration d'Agents")
print("=" * 60)

# Afficher chaque tour
turns = [
    ("❓ Q1 - Introduction", q1),
    ("💡 R1 - Définition", r1),
    ("❓ Q2 - Patterns", q2),
    ("💡 R2 - Exemples", r2),
    ("❓ Q3 - Futur", q3),
    ("💡 R3 - Vision", r3),
]

for label, task in turns:
    print(f"\n{label}:")
    print(f"   {task.output.raw if task.output else 'N/A'}")
    print("-" * 40)

print("\n" + "=" * 60)
print("📋 RÉSUMÉ DE L'INTERVIEW")
print("=" * 60)
print(f"\n{resume.output.raw if resume.output else result}")
print("\n" + "=" * 60)
print("✅ Concepts démontrés: Single-turn, Rôles, Multi-turn, Résumé")
print("=" * 60)
