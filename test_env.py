import os
from dotenv import load_dotenv

load_dotenv(override=True)

print("SUPABASE_URL:", repr(os.getenv("SUPABASE_URL")))
print("SUPABASE_SERVICE_ROLE_KEY set:", bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY")))
print("MISTRAL_API_KEY set:", bool(os.getenv("MISTRAL_API_KEY")))