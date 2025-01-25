import logging
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, validator

# Load environment variables from the .env file
load_dotenv(dotenv_path="./.env")


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,  # Use DEBUG for detailed logs or INFO for general use
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Logging is configured.")


class LLMSettings(BaseModel):
    """Base settings for the Language Model."""
    temperature: float = Field(default=0.0, description="Control response randomness.")
    max_tokens: Optional[int] = Field(default=None, description="Max token limit.")
    max_retries: int = Field(default=3, description="Number of retries on failure.")

    @validator("temperature")
    def validate_temperature(cls, value):
        if not (0.0 <= value <= 1.0):
            raise ValueError("Temperature must be between 0.0 and 1.0.")
        return value


class GeminiSettings(LLMSettings):
    """Gemini-specific settings extending LLMSettings."""
    api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY"),
        description="API key for accessing the Gemini platform."
    )
    default_model: str = Field(
        default="gemini-1.5-flash",
        description="Default Gemini model for text generation."
    )
    embedding_model: str = Field(
        default="models/text-embedding-004",
        description="Gemini embedding model for vector generation."
    )

    @validator("api_key")
    def validate_api_key(cls, value):
        if not value:
            raise ValueError("Gemini API key must be provided.")
        return value

TIMESCALE_SERVICE_URL = "postgresql://postgres:password@localhost:5432/postgres"
class DatabaseSettings(BaseModel):
    """Settings for database connection."""
    service_url: str = Field(
        # default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"),
        default_factory=lambda: TIMESCALE_SERVICE_URL,
        description="Connection URL for TimescaleDB."
    )

    @validator("service_url")
    def validate_service_url(cls, value):
        if not value:
            raise ValueError("Database service URL must be provided.")
        return value


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""
    table_name: str = Field(
        default="contract_embeddings",
        description="Table name for storing embeddings."
    )
    embedding_dimensions: int = Field(
        # default=1536,
        default=768, 
        description="Dimension size of the embedding vectors."
    )
    time_partition_interval: timedelta = Field(
        default=timedelta(days=7),
        description="Interval for time-based partitioning of embeddings."
    )
    similarity_threshold: float = Field(
        default=0.8,
        description="Minimum similarity score to consider a match relevant."
    )

    @validator("similarity_threshold")
    def validate_similarity_threshold(cls, value):
        if not (0.0 <= value <= 1.0):
            raise ValueError("Similarity threshold must be between 0.0 and 1.0.")
        return value


class FileSettings(BaseModel):
    """Settings for handling uploaded contract files."""
    upload_directory: str = Field(
        default="./uploads",
        description="Directory for storing uploaded contract files."
    )
    allowed_file_types: list[str] = Field(
        default=["csv", "pdf", "txt"],
        description="Permitted file types for uploads."
    )
    max_file_size: int = Field(
        default=10485760,  # 10 MB
        description="Maximum allowed file size in bytes."
    )
    preprocess_text: bool = Field(
        default=True,
        description="Enable preprocessing for contract text."
    )

    @validator("allowed_file_types", each_item=True)
    def validate_file_types(cls, value):
        if value not in ["csv", "pdf", "txt"]:
            raise ValueError(f"Unsupported file type: {value}")
        return value


class Settings(BaseModel):
    """Main settings combining all configurations."""
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    files: FileSettings = Field(default_factory=FileSettings)

import json  # Add this import at the top

import json
from datetime import timedelta

@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    try:
        settings = Settings()
        setup_logging()
        logging.info("Settings initialized successfully.")

        # Custom serialization for non-JSON serializable fields
        def custom_serializer(obj):
            if isinstance(obj, timedelta):
                return str(obj)  # Convert timedelta to string
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        logging.debug(f"Loaded settings: {json.dumps(settings.dict(), indent=2, default=custom_serializer)}")
        return settings
    except ValidationError as e:
        logging.error(f"Error in settings initialization: {e}")
        raise

