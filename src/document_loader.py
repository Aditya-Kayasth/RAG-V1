import os
from typing import List, Dict, Any
from pypdf import PdfReader

class DocumentLoader:
    """Loads text, markdown, and PDF files and splits them into chunks for vector indexing."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: str) -> str:
        """Reads text content from .txt, .md, or .pdf files."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.txt', '.md', '.markdown']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext == '.pdf':
            reader = PdfReader(file_path)
            text_pages = []
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
            return "\n\n".join(text_pages)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: .txt, .md, .pdf")

    def split_text(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Splits a document text into overlapping chunks with metadata."""
        chunks = []
        start = 0
        text_length = len(text)
        chunk_id = 0

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "id": f"{os.path.basename(source)}_chunk_{chunk_id}",
                    "text": chunk_text,
                    "metadata": {
                        "source": source,
                        "chunk_index": chunk_id,
                        "character_count": len(chunk_text)
                    }
                })
                chunk_id += 1
            
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads and splits a file into indexed document chunks."""
        text = self.load_file(file_path)
        return self.split_text(text, source=file_path)
