from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import chat, auth, profile, health, history, insights

app = FastAPI(title="MedXAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(profile.router, prefix="/api/v1", tags=["profile"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(history.router, prefix="/api/v1", tags=["history"])
app.include_router(insights.router, prefix="/api/v1", tags=["insights"])

@app.get("/")
def root():
    return {"status": "MedXAI backend is live "}