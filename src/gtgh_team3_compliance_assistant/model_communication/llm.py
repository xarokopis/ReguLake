import os

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()


class ChatLLM:
    def __init__(self):
        self.client = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.7,
        )

    def generate(self, question: str, context: str) -> str:
        prompt = f"""Use only the provided context to answer the question.
If the answer is not present in the context, say: "The document does not contain enough information."

Question:
{question}

Context:
{context}"""

        response = self.client.invoke(
            [
                SystemMessage(content="You answer only from the retrieved document context."),
                HumanMessage(content=prompt),
            ]
        )

        return response.content
