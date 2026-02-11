from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=180
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful teaching assistant. Explain clearly and briefly."),
    ("human", "Explain prompt engineering.")
])

messages = prompt.format_messages()
response = llm.invoke(messages)

print(response.content)
