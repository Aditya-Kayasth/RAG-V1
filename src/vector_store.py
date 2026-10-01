import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from src.config import Config
from src.embeddings import GeminiEmbeddingFunction

class VectorStoreManager:
    """Manages persistent ChromaDB collections, vector embeddings, and similarity queries."""

    def __init__(self, collection_name: str = None, persist_dir: str = None):
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.persist_dir = persist_dir or Config.CHROMA_DB_DIR
        
        # Ensure persistence directory exists
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = GeminiEmbeddingFunction()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG Document Collection"}
        )

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Generates embeddings and stores document chunks in ChromaDB."""
        if not chunks:
            return

        ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        # Compute embeddings using Gemini API
        embeddings = self.embedding_fn.embed_documents(texts)

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query(self, query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Queries ChromaDB for the top-k most relevant document chunks."""
        query_embedding = self.embedding_fn.embed_query(query_text)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        matched_chunks = []
        if results and results.get("documents") and len(results["documents"][0]) > 0:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(documents)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(documents)

            for doc, meta, dist in zip(documents, metadatas, distances):
                matched_chunks.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                })

        return matched_chunks

    def get_count(self) -> int:
        """Returns total document chunks in collection."""
        return self.collection.count()
