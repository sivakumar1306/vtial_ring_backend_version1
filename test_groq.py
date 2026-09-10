import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

key = os.getenv("GROQ_API_KEY") or os.getenv("MISTRAL_API_KEY")
model = "openai/gpt-oss-120b"

print(f"Testing ChatGroq with model '{model}'...")
print("Key loaded:", bool(key), "starts with:", key[:6] if key else None)

if not key:
    print("Error: GROQ_API_KEY not found in environment.")
else:
    try:
        llm = ChatGroq(api_key=key, model=model, temperature=0.1)
        response = llm.invoke("Say hello in one word")
        print("SUCCESS! Response:", response.content.strip())
    except Exception as e:
        print("Error invoking ChatGroq:", str(e))