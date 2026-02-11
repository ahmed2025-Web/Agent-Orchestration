from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=220
)

template = PromptTemplate.from_template(
    "You are an AI tutor.\n"
    "In this question, RAG means Retrieval-Augmented Generation (LLM + document retrieval), not Red-Amber-Green.\n"
    "Answer in French, in 5 lines max:\n"
    "Question: {question}"
)

question = "What is RAG and why is it useful?"
prompt = template.format(question=question)

response = llm.invoke(prompt)
print(response.content)
