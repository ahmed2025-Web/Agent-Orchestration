"""
AI STUDY PARTNER (L'Assistant d'Étude Intelligent)
"""

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
import os

# 1. Configuration initiale
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("MISTRAL_API_KEY") or "not_used"

llm = LLM(
    model="mistral/mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

# 2. Outil de recherche dans le PDF 
pdf_tool = PDFSearchTool(
    pdf='3.VectorDatabases/diapo.pdf',
    collection_name='diapo_rag',
    config={
        "embedding_model": {
            "provider": "sentence-transformer",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            }
        }
    }
)

# 3. Defining Agents (The Pedagogical Team)
archivist = Agent(
    role="Scientific Archivist",
    goal="Extract precise and technical information from the PDF document.",
    backstory="""You are an expert in documentary research. Your role is to search 
    through the archives to find the 'raw truth' on a given subject. 
    You do not simplify anything; you report the facts exactly as they are written.""",
    tools=[pdf_tool],
    llm=llm,
    verbose=True
)

pedagogue = Agent(
    role="Educational Simplifier",
    goal="Simplify complex concepts to make them accessible to everyone.",
    backstory="""You are an exceptional teacher. You take the technical notes 
    from the archivist and transform them into a clear, illustrated 
    explanation that is easy to remember. You use analogies if necessary.""",
    llm=llm,
    verbose=True
)

quiz_master = Agent(
    role="Quiz Designer",
    goal="Create an assessment test to verify the student's understanding.",
    backstory="""You are an expert in educational assessment. You create 
    relevant questions that target the essential points of the subject to ensure 
    that the student has correctly understood the teacher's explanation.""",
    llm=llm,
    verbose=True
)

def create_study_guide(topic):
    search_task = Task(
        description=f"Search for everything the PDF says about the following topic: {topic}",
        agent=archivist,
        expected_output="A detailed technical report containing quotes from the PDF."
    )

    summary_task = Task(
        description=f"Based on the technical research, explain the concept of {topic} in simple terms.",
        agent=pedagogue,
        context=[search_task],
        expected_output="A clear, structured, and pedagogical explanation in English."
    )

    output_filename = f"3.VectorDatabases/study_guide_{topic.replace(' ', '_')}.md"

    quiz_task = Task(
        description=f"""Create a complete study guide for {topic}. 
        The document must contain:
        1. A clear pedagogical explanation.
        2. A quiz of 3 questions (with answers at the end).
        """,
        agent=quiz_master,
        context=[summary_task],
        expected_output="A structured Markdown document with titles and paragraphs.",
        output_file=output_filename 
    )

    # 5. Création et lancement du Crew
    study_crew = Crew(
        agents=[archivist, pedagogue, quiz_master],
        tasks=[search_task, summary_task, quiz_task],
        process=Process.sequential
    )

    print(f"\n🎓 Generating study guide for: {topic}...")
    result = study_crew.kickoff()
    print(f"\n✅ Your study guide has been saved to: {output_filename}")
    return result

if __name__ == "__main__":
    print("=" * 60)
    print("🌟 WELCOME TO YOUR AI STUDY PARTNER")
    print("=" * 60)
    
    sujet = input("👉 Which concept from the PDF would you like to study today? ")
    
    if sujet.strip():
        resultat = create_study_guide(sujet)
        
        print("\n" + "📚" * 20)
        print("YOUR PERSONALIZED STUDY GUIDE")
        print("📚" * 20)
        print(resultat)
        print("=" * 60)
