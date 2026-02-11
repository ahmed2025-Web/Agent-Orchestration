from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=150
)

template = PromptTemplate.from_template(
    "Explain {topic} in simple terms."
)

prompt = template.format(topic="vector databases")

response = llm.invoke(prompt)

print(response.content)
