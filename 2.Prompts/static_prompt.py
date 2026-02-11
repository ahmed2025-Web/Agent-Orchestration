from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=120
)

prompt = "Explain what a Large Language Model is in one sentence."
response = llm.invoke(prompt)

print(response.content)
