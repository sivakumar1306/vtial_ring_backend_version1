import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

key = os.getenv("MISTRAL_API_KEY")
print("Key loaded:", bool(key), "starts with:", key[:6] if key else None)

llm = ChatMistralAI(api_key=key, model="mistral-large-latest")
response = llm.invoke("Say hello in one word")
print("Response:", response.content)