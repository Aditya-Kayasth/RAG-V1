import os
import sys
import argparse
from src.config import Config
from src.document_loader import DocumentLoader
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline

def ingest_command(path: str):
    """Ingests a file or directory into ChromaDB vector store."""
    if not os.path.exists(path):
        print(f"Error: Path '{path}' does not exist.")
        sys.exit(1)

    loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
    store = VectorStoreManager()

    files_to_process = []
    if os.path.isfile(path):
        files_to_process.append(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(('.txt', '.md', '.pdf')):
                    files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        print("No supported files (.txt, .md, .pdf) found to process.")
        return

    total_chunks = 0
    print(f"Processing {len(files_to_process)} file(s)...")
    for file_path in files_to_process:
        print(f" -> Loading: {file_path}")
        chunks = loader.process_file(file_path)
        store.add_documents(chunks)
        total_chunks += len(chunks)

    print(f"Success! Ingested {total_chunks} chunk(s) into ChromaDB.")
    print(f"Total chunks in vector collection: {store.get_count()}")

def query_command(question: str, top_k: int = 4):
    """Queries RAG pipeline with user question."""
    print(f"\nQuestion: {question}")
    print("Retrieving context & generating answer with Gemini...\n")
    
    pipeline = RAGPipeline()
    result = pipeline.generate_answer(question, top_k=top_k)

    print("=== ANSWER ===")
    print(result["answer"])
    print("\n=== RETRIEVED SOURCES ===")
    for idx, chunk in enumerate(result["context"], 1):
        source = chunk["metadata"].get("source", "Unknown")
        dist = chunk.get("distance", 0.0)
        print(f" [{idx}] Source: {source} (distance: {dist:.4f})")
        print(f"     Snippet: {chunk['text'][:150]}...\n")

def stats_command():
    """Displays information about the vector collection."""
    store = VectorStoreManager()
    count = store.get_count()
    print("=== ChromaDB Collection Stats ===")
    print(f"Collection Name: {Config.COLLECTION_NAME}")
    print(f"Storage Path   : {os.path.abspath(Config.CHROMA_DB_DIR)}")
    print(f"Total Documents: {count}")

def main():
    parser = argparse.ArgumentParser(description="Gemini RAG System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest document or directory into ChromaDB")
    ingest_parser.add_argument("path", type=str, help="File path or directory path to ingest")

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Ask a question against stored documents")
    query_parser.add_argument("question", type=str, help="Question to ask")
    query_parser.add_argument("--top_k", type=int, default=4, help="Number of context chunks to retrieve")

    # Stats subcommand
    subparsers.add_parser("stats", help="View vector database status")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_command(args.path)
    elif args.command == "query":
        query_command(args.question, top_k=args.top_k)
    elif args.command == "stats":
        stats_command()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
