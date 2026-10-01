import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Config:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "chromadb_store")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_documents")

    @classmethod
    def validate(cls):
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")
