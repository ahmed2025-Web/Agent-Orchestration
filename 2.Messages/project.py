"""
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
    role="Tech Journalist",
    goal="Ask relevant and progressive questions about AI agent orchestration.",
    backstory="""You are a technology journalist for a specialized magazine.
    You ask clear questions that range from simple to complex.
    You address a technical audience, but not necessarily experts.""",
    llm=llm
)

expert = Agent(
    role="AI Architect",
    goal="Explain agent orchestration in a technical but accessible way.",
    backstory="""You are a senior AI architect who designs multi-agent systems.
    You provide structured answers with concrete examples.
    You use analogies where helpful.""",
    llm=llm
)

summarizer = Agent(
    role="Editor-in-Chief",
    goal="Synthesize an interview into a structured and engaging summary.",
    backstory="""You are the editor-in-chief of a tech magazine.
    You know how to extract the key points of a conversation.
    You produce clear summaries with essential takeaways.""",
    llm=llm
)

# ── Tasks (Interview in 3 rounds + summary) ─────────────────

# Q1: Introduction question
q1 = Task(
    description="Ask an introductory question about what AI agent orchestration is and why it's important today.",
    agent=interviewer,
    expected_output="A clear and engaging introductory question."
)

# R1: Answer to the introductory question
r1 = Task(
    description="Answer the journalist's question about AI agent orchestration.",
    agent=expert,
    expected_output="A structured answer with a definition and its importance.",
    context=[q1]
)

# Q2: Question on patterns
q2 = Task(
    description="Based on the response, ask a question about different orchestration patterns (sequential, parallel, conditional).",
    agent=interviewer,
    expected_output="A technical question about orchestration patterns.",
    context=[r1]
)

# R2: Answer on patterns
r2 = Task(
    description="Explain orchestration patterns with concrete examples for each.",
    agent=expert,
    expected_output="An explanation of patterns with examples.",
    context=[q2]
)

# Q3: Question on the future
q3 = Task(
    description="Ask a final question about the challenges and the future of agent orchestration.",
    agent=interviewer,
    expected_output="A forward-looking question.",
    context=[r2]
)

# R3: Answer on the future
r3 = Task(
    description="Share your vision regarding the current challenges and the future of agent orchestration.",
    agent=expert,
    expected_output="A forward-looking vision covering challenges and opportunities.",
    context=[q3]
)

# Final summary
summary = Task(
    description="""Synthesize the entire interview into a structured summary that includes:
    - A catchy title
    - 3 key points from the interview
    - A striking quote from the expert
    - A concluding sentence""",
    agent=summarizer,
    expected_output="A structured summary of the interview.",
    context=[q1, r1, q2, r2, q3, r3]
)

# ── Execution ───────────────────────────────────────────────

crew = Crew(
    agents=[interviewer, expert, summarizer],
    tasks=[q1, r1, q2, r2, q3, r3, summary],
    process=Process.sequential,
    verbose=False
)

result = crew.kickoff()

print("=" * 60)
print("🎙️  PROJECT: AI INTERVIEW - Agent Orchestration")
print("=" * 60)

# Display each round
turns = [
    ("❓ Q1 - Introduction", q1),
    ("💡 R1 - Definition", r1),
    ("❓ Q2 - Patterns", q2),
    ("💡 R2 - Examples", r2),
    ("❓ Q3 - Future", q3),
    ("💡 R3 - Vision", r3),
]

for label, task in turns:
    print(f"\n{label}:")
    print(f"   {task.output.raw if task.output else 'N/A'}")
    print("-" * 40)

print("\n" + "=" * 60)
print("📋 INTERVIEW SUMMARY")
print("=" * 60)
print(f"\n{summary.output.raw if summary.output else result}")
print("\n" + "=" * 60)
print("✅ Concepts demonstrated: Single-turn, Roles, Multi-turn, Summary")
print("=" * 60)
