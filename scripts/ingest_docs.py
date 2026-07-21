from sentence_transformers import SentenceTransformer
from db.supabase import supabase
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def ingest_file(path, source_name):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = chunk_text(text)
    embeddings = model.encode(chunks).tolist()
    rows = [{"source": source_name, "content": c, "embedding": e}
            for c, e in zip(chunks, embeddings)]
    supabase.table("medical_documents").insert(rows).execute()
    print(f"Ingested {len(rows)} chunks from {source_name}")

if __name__ == "__main__":
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "data", "medical_docs")
    for fname in os.listdir(docs_dir):
        ingest_file(os.path.join(docs_dir, fname), fname)