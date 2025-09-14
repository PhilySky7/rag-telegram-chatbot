from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM configs
    MODEL_NAME: str = "sonar"
    MODEL_PROVIDER: str = "perplexity"
    MODEL_TEMPERATURE: float = 0.4

    # Embedding model
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"

    # Vector store configs
    PERSIST_DIRECTORY: str = "./data/chroma"
    COLLECTION_NAME: str = "rag_collection"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

settings = Settings()
