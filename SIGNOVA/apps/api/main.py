"""
Main FastAPI entrypoint for SIGNOVA.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apps.api.routes.health import router as health_router
from apps.api.routes.live import router as live_router

app = FastAPI(
    title="SIGNOVA API",
    description="Continuous Indian Sign Language (ISL) -> English Translation System API",
    version="0.1.0",
)

# Enable CORS for frontend local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(health_router)
app.include_router(live_router)


@app.get("/")
async def root():
    return {
        "message": "Welcome to SIGNOVA API",
        "phase": 0,
        "docs_url": "/docs",
        "health_url": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.api.main:app", host="127.0.0.1", port=8000, reload=True)
