from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Base settings class for configuration management."""

    # Model configurations for llm
    MODEL_NAME: str = "sonar"
    MODEL_PROVIDER: str = "perplexity"
    MODEL_TEMPERATURE: float = 0.7

    # Embedding model
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"

    # Vector store configuration
    PERSIST_DIRECTORY: str = "./chroma_store"
    COLLECTION_NAME: str = "rag_collection"

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
settings = Settings()
