import warnings
import logging
warnings.simplefilter("ignore")
logging.basicConfig(level=logging.ERROR)

from typing import Dict, Any, List
from google import genai
from src.config import Config
from src.vector_store import VectorStoreManager



class RAGPipeline:
    """Combines document retrieval from ChromaDB with Google Gemini LLM generation."""

    def __init__(self, model_name: str = None):
        Config.validate()
        self.model_name = model_name or Config.GEMINI_MODEL
        self.vector_store = VectorStoreManager()
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)

    def generate_answer(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """Retrieves relevant context chunks and asks Gemini to answer the question."""
        # Step 1: Retrieve relevant context chunks from ChromaDB
        relevant_chunks = self.vector_store.query(question, top_k=top_k)

        if not relevant_chunks:
            return {
                "question": question,
                "answer": "No relevant documents found in the vector store to answer your query.",
                "context": []
            }

        # Format context for prompt
        context_str = "\n\n---\n\n".join(
            [f"[Source: {chunk['metadata'].get('source', 'Unknown')}]\n{chunk['text']}" for chunk in relevant_chunks]
        )

        # Construct prompt
        prompt = f"""You are a helpful and precise AI assistant. Answer the user's question using ONLY the provided context snippets below.
If the context does not contain enough information to answer the question, state that clearly.

=== CONTEXT ===
{context_str}

=== USER QUESTION ===
{question}

=== ANSWER ===
"""

        # Step 2: Call Gemini LLM using model specified in config with graceful fallback
        candidate_models = [self.model_name]
        if not self.model_name.startswith("models/"):
            candidate_models.append(f"models/{self.model_name}")
        candidate_models.extend([
            "gemini-2.5-flash", "models/gemini-2.5-flash",
            "gemini-2.0-flash", "models/gemini-2.0-flash",
            "gemini-1.5-flash", "models/gemini-1.5-flash"
        ])

        answer_text = None
        last_error = None

        for model_id in candidate_models:
            try:
                response = self.client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                )
                answer_text = response.text if hasattr(response, 'text') else str(response)
                break
            except Exception as e:
                last_error = e

        if not answer_text:
            raise RuntimeError(f"Failed to generate content with Gemini API. Error: {last_error}")

        return {
            "question": question,
            "answer": answer_text,
            "context": relevant_chunks
        }

