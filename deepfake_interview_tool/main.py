"""
Deepfake Interview Alert Tool - Main Application
Phase 1 + Phase 2 Implementation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from contextlib import asynccontextmanager

# Import routers
from routers import interview_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Deepfake Interview Alert Tool")
    yield
    # Shutdown
    logger.info("Shutting down Deepfake Interview Alert Tool")

# Create FastAPI app
app = FastAPI(
    title="Deepfake Interview Alert Tool",
    description="AI-powered tool to detect potential deepfake indicators in virtual interviews",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(interview_router, prefix="/api/v1", tags=["interview"])

@app.get("/")
async def root():
    return {
        "message": "Deepfake Interview Alert Tool API",
        "version": "2.0.0",
        "phase": "Phase 1 + Phase 2 implemented",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "deepfake-interview-tool"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)