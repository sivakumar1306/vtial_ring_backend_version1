import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

key = os.getenv("GROQ_API_KEY") or os.getenv("MISTRAL_API_KEY")
print("Key loaded:", bool(key), "starts with:", key[:6] if key else None)

if not key:
    print("Error: Neither GROQ_API_KEY nor MISTRAL_API_KEY found in environment.")
else:
    llm = ChatGroq(api_key=key, model="llama-3.3-70b-versatile", temperature=0.1)
    try:
        response = llm.invoke("Say hello in one word")
        print("Response:", response.content)
    except Exception as e:
        print("Error invoking Groq:", str(e))
