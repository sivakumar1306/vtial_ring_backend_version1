import asyncio
from sentence_transformers import SentenceTransformer
from db.supabase import supabase
from services.medlineplus import fetch_medlineplus

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

SIMILARITY_THRESHOLD = 0.45

CLINICAL_KEYWORDS = [
    "what is", "what are", "symptoms", "causes", "treatment", "cure",
    "diagnosis", "medicine", "medication", "disease", "condition", "disorder",
    "infection", "virus", "bacteria", "chronic", "acute", "prevent", "prevention",
    "risk", "complication", "side effect", "drug", "dose", "therapy", "remedy",
    "diabetes", "hypertension", "anxiety", "cold", "flu", "blood pressure",
    "heart", "sleep disorder", "deficiency", "allergy", "cancer", "stroke",
    "fever", "headache", "migraine", "asthma", "depression", "thyroid",
    "anemia", "arthritis", "kidney", "liver", "obesity", "cholesterol"
]

BIOMETRIC_KEYWORDS = [
    "my heart rate", "my sleep", "my steps", "my spo2", "my hrv",
    "my stress", "my calories", "my data", "my health", "my ring",
    "yesterday", "last night", "today", "this week", "my score",
    "how am i", "how did i", "my readings", "my levels"
]

def classify_intent(message: str) -> str:
    msg = message.lower()
    is_clinical = any(kw in msg for kw in CLINICAL_KEYWORDS)
    is_biometric = any(kw in msg for kw in BIOMETRIC_KEYWORDS)
    if is_clinical and is_biometric:
        return "both"
    elif is_clinical:
        return "clinical"
    elif is_biometric:
        return "biometric"
    else:
        return "general"

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def ingest_content(content: str, source: str) -> int:
    """Embed and store new content into Supabase. Returns number of chunks ingested."""
    try:
        chunks = chunk_text(content)
        embeddings = get_embedding_model().encode(chunks).tolist()
        rows = [
            {"source": source, "content": c, "embedding": e}
            for c, e in zip(chunks, embeddings)
        ]
        supabase.table("medical_documents").insert(rows).execute()
        print(f"[RAG] Ingested {len(rows)} new chunks from {source}")
        return len(rows)
    except Exception as e:
        print(f"[RAG] Ingest failed for {source}: {e}")
        return 0

def source_already_ingested(source: str) -> bool:
    """Check if a source is already in the knowledge base."""
    try:
        result = supabase.table("medical_documents")\
            .select("id")\
            .ilike("source", f"%{source}%")\
            .limit(1)\
            .execute()
        return len(result.data) > 0
    except:
        return False

def retrieve_context(query: str, match_count: int = 5) -> list:
    query_embedding = get_embedding_model().encode(query).tolist()
    result = supabase.rpc("match_medical_documents", {
        "query_embedding": query_embedding,
        "match_count": match_count
    }).execute()
    return result.data

async def retrieve_context_with_expansion(query: str, match_count: int = 5) -> list:
    """
    Retrieve context and return it immediately. If the top match's similarity
    is below threshold, kick off a BACKGROUND task that fetches fresh content
    from MedlinePlus, embeds it, and ingests it into the knowledge base for
    FUTURE queries — but this current call does not wait on any of that.

    Previously, a low-similarity match triggered a live MedlinePlus API call,
    embedding-model encoding of the fetched content, a Supabase insert, and a
    second re-embed + re-query — ALL synchronously in the /chat request's
    critical path. On a cold SentenceTransformer load (first request after a
    deploy/restart) or a query the knowledge base didn't already cover well
    (i.e. most casual/novel phrasing), this alone added many seconds —
    sometimes 20+ — to the user's response time. Fire-and-forgetting the
    expansion means the user gets today's best available answer immediately,
    while the knowledge base still improves for next time.
    """
    results = retrieve_context(query, match_count)

    if results and results[0].get("similarity", 0) >= SIMILARITY_THRESHOLD:
        print(f"[RAG] Good coverage (sim: {round(results[0]['similarity'], 3)}) — using existing knowledge base")
        return results

    print(f"[RAG] Low coverage — scheduling background MedlinePlus fetch for: '{query}'")
    # Fire-and-forget: asyncio.create_task schedules this to run concurrently
    # without the caller awaiting it, so it can't add latency to this response.
    # A reference is kept only to avoid the task being garbage-collected
    # mid-flight (a known asyncio gotcha with un-referenced create_task calls).
    task = asyncio.create_task(_expand_knowledge_base(query))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    # Return whatever we have right now — even a below-threshold match beats
    # making the user wait 10-20+ seconds for a "perfect" one that may not
    # even end up better after expansion.
    return results


# Keeps strong references to in-flight background tasks so they aren't
# garbage-collected before completing (asyncio only holds a weak reference
# to tasks created via create_task unless something else references them).
_background_tasks: set[asyncio.Task] = set()


async def _expand_knowledge_base(query: str):
    """Runs after the /chat response has already been sent — improves the
    knowledge base for future queries, never blocks the current one."""
    try:
        fetched = await fetch_medlineplus(query, max_results=3)
        newly_ingested = 0
        for item in fetched:
            source_key = item['title'].lower().replace(' ', '_')[:40]
            if not source_already_ingested(source_key):
                newly_ingested += ingest_content(item['content'], source_key)
            else:
                print(f"[RAG] Already have: {source_key}")
        if newly_ingested > 0:
            print(f"[RAG] Background expansion complete — {newly_ingested} new chunk(s) ingested for '{query}'")
        else:
            print(f"[RAG] Background expansion found nothing new for '{query}'")
    except Exception as e:
        print(f"[RAG] Background expansion failed for '{query}': {e}")