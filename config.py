import os
from typing import List, Optional
from pydantic import validator
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """Application settings configured via environment variables"""
    
    # API Keys
    OPENAI_API_KEY: str
    HUGGINGFACE_TOKEN: str = "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"
    
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
    
    @validator("OPENAI_API_KEY")
    def validate_openai_key(cls, v):
        if not v or v == "your-openai-api-key-here":
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set this variable with your OpenAI API key."
            )
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Initialize settings
settings = Settings()

# Ensure storage directory exists
os.makedirs(settings.STORAGE_DIR, exist_ok=True)

# Set OpenAI API key in environment for modules that expect it there
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

def get_settings():
    """Function to get settings (can be used as dependency in FastAPI)"""
    return settings