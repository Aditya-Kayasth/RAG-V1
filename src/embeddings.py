from typing import List
from google import genai
from src.config import Config

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

class GeminiEmbeddingFunction:
    """Wrapper for generating vector embeddings using Google Gemini API with fallback to local ChromaDB embeddings."""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        Config.validate()
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.model_name = model_name or Config.GEMINI_EMBEDDING_MODEL
        self.client = genai.Client(api_key=self.api_key)
        self._local_fallback = None

    def _get_local_fallback(self):
        if self._local_fallback is None:
            self._local_fallback = DefaultEmbeddingFunction()
        return self._local_fallback

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a list of document strings."""
        if not texts:
            return []

        candidates = [self.model_name]
        if not self.model_name.startswith("models/"):
            candidates.append(f"models/{self.model_name}")
        candidates.extend(["text-embedding-004", "models/text-embedding-004", "embedding-001", "models/embedding-001"])

        for model in candidates:
            try:
                embeddings = []
                for text in texts:
                    response = self.client.models.embed_content(
                        model=model,
                        contents=text,
                    )
                    if hasattr(response, 'embedding') and hasattr(response.embedding, 'values'):
                        embeddings.append(response.embedding.values)
                    elif hasattr(response, 'embeddings') and len(response.embeddings) > 0:
                        embeddings.append(response.embeddings[0].values)
                    else:
                        raise ValueError(f"Could not extract embeddings from response: {response}")
                return embeddings
            except Exception:
                continue

        # If Gemini embedding API fails, use local embedding function
        local_fn = self._get_local_fallback()
        return local_fn(texts)

    def embed_query(self, query: str) -> List[float]:
        """Generates an embedding vector for a single search query."""
        results = self.embed_documents([query])
        return results[0] if results else []

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

