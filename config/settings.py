from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    APP_NAME: str = "Multimedia RAG Assistant"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
    CHUNKS_DIR: str = os.path.join(DATA_DIR, "processed", "chunks")
    METADATA_DIR: str = os.path.join(DATA_DIR, "processed", "metadata")
    EMBEDDINGS_DIR: str = os.path.join(DATA_DIR, "processed", "embeddings")
    
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    HUGGINGFACE_API_KEY: Optional[str] = None

    LOG_DIR: str = "logs"
    LOG_LEVEL: str = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=True
    )
    
settings = Settings()