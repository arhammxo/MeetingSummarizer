import os
from typing import List, Optional
from pydantic import validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """Application settings configured via environment variables"""
    
    # LLM Configuration
    LLM_PROVIDER: str = "ollama"  # Options: "openai", "ollama"
    
    # OpenAI Configuration (legacy, can be removed if fully migrating)
    OPENAI_API_KEY: Optional[str] = None
    
    # Ollama Configuration
    OLLAMA_API_BASE: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.3:70b"  # Default model
    OLLAMA_SUMMARIZATION_MODEL: str = "llama3.3:70b"  # For summarization
    OLLAMA_MULTILINGUAL_MODEL: str = "llama3.3:70b"  # For multilingual support
    
    # Audio Processing
    HUGGINGFACE_TOKEN: Optional[str] = None
    
    # Storage
    STORAGE_DIR: str = "job_results"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"
    WORKERS: int = 1
    
    # Security
    ENABLE_HTTPS: bool = False
    SSL_CERT_PATH: Optional[str] = None
    SSL_KEY_PATH: Optional[str] = None
    
    # CORS
    CORS_ORIGINS: str = "*"
    
    @validator("CORS_ORIGINS")
    def parse_cors_origins(cls, v):
        if v == "*":
            return ["*"]
        return [i.strip() for i in v.split(",") if i.strip()]
    
    @validator("LLM_PROVIDER")
    def validate_llm_provider(cls, v, values):
        if v not in ["openai", "ollama"]:
            raise ValueError(f"LLM_PROVIDER must be 'openai' or 'ollama', got {v}")
        
        # If using OpenAI, validate API key
        if v == "openai" and not values.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is required when LLM_PROVIDER is 'openai'"
            )
            
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Initialize settings
settings = Settings()

# Ensure storage directory exists
os.makedirs(settings.STORAGE_DIR, exist_ok=True)

# Set OpenAI API key in environment if using OpenAI
if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

def get_settings():
    """Function to get settings (can be used as dependency in FastAPI)"""
    return settings