# Gemini RAG System (v1 Prototype)

A lightweight, modular, and production-ready **Retrieval-Augmented Generation (RAG)** prototype built with **Google Gemini LLM**, **Gemini Embeddings**, and **ChromaDB**.

> [!NOTE]
> This repository serves as a personal trial and reference implementation for exploring RAG workflows using Google Gemini models and local vector database storage.

---

## 📌 Features

- **Embedded Local Vector Store**: Uses **ChromaDB** to index and store document chunk embeddings persistently on disk without needing a separate database server.
- **Google GenAI SDK Integration**: Connects to Google's latest Gemini models (e.g. `gemini-3.5-flash`) for context-grounded response generation.
- **Robust Embedding Pipeline**: Generates high-dimensional vector embeddings with automated fallback to local ONNX embeddings (`all-MiniLM-L6-v2`) for offline reliability.
- **Multi-Format Document Ingestion**: Parses `.txt`, `.md`, and `.pdf` files into overlapping chunk snippets.
- **Command Line Interface (CLI)**: Simple `main.py` entrypoint for document ingestion, querying, and collection statistics.

---

## 🏗 Architecture Overview

```
RAG-V1/
├── .env.example          # Environment variables template
├── .gitignore            # Excluded files (.venv, .env, chromadb_store)
├── requirements.txt      # Project dependencies
├── main.py               # Main CLI application
├── sample_data/          # Sample documents directory
│   └── sample.txt        # Initial dataset
└── src/
    ├── config.py         # Environment validator & settings
    ├── document_loader.py# Multi-format document parser & text chunker
    ├── embeddings.py     # Gemini embedding function with local fallback
    ├── vector_store.py   # ChromaDB manager (upsert, query, count)
    └── rag_pipeline.py   # RAG pipeline coordinating retrieval + Gemini LLM
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key ([Get an API Key](https://aistudio.google.com/))

### 2. Environment Setup
Clone the repository and set up a virtual environment:

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the project root (copied from `.env.example`):

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
GEMINI_EMBEDDING_MODEL=text-embedding-004
CHROMA_DB_DIR=chromadb_store
```

---

## 💻 Usage

### 1. Ingest Documents into Vector Database
Ingest a single document or an entire directory of files (`.txt`, `.md`, `.pdf`):

```powershell
python main.py ingest sample_data/sample.txt
```

### 2. Query the Knowledge Base
Ask questions grounded in the ingested documents:

```powershell
python main.py query "Who is the project lead?"
```

**Example Output:**
```text
Question: Who is the project lead?
Retrieving context & generating answer with Gemini...

=== ANSWER ===
Based on the provided context, the project lead is Ram Rajesh Shrivastav.

=== RETRIEVED SOURCES ===
 [1] Source: sample_data/sample.txt (distance: 1.8945)
```

### 3. Inspect Collection Statistics
Check total ingested documents and storage paths:

```powershell
python main.py stats
```

---

## 📄 License
This project is open-source under the MIT License.
