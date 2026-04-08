# add_document.py

import os
import re
from dotenv import load_dotenv
from langchain_core.documents import Document
from pypdf import PdfReader  # Add this import
from vector_store import vector_store

load_dotenv()

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts all text from a PDF file."""
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def chunk_projects(text: str) -> list[Document]:
    """Chunks text by individual projects based on 'Project X:' pattern."""
    documents = []
    
    # Positive lookahead to split right before "Project X:"
    # Note: Added \s* to handle potential extra spaces in PDF extraction
    project_pattern = r'(?=Project\s+\d+:)'
    
    parts = re.split(project_pattern, text)
    
    for part in parts:
        part = part.strip()
        
        # Filter out headers/extra PDF text and small fragments
        if not part.startswith('Project') or len(part) < 50:
            continue
            
        # Clean up common PDF extraction artifacts (multiple newlines, etc.)
        clean_content = re.sub(r'\n\s*\n', '\n', part)
        
        lines = [line.strip() for line in clean_content.split('\n') if line.strip()]
        project_title = lines[0] if lines else "Unknown Project"
        
        # Basic category tagging
        category = "General"
        if any(word in clean_content.upper() for word in ["AI", "GENERATIVE", "LLM"]):
            category = "AI/ML"
        elif "AWS" in clean_content.upper() or "LAMBDA" in clean_content.upper():
            category = "Serverless/Backend"
            
        metadata = {
            "source": "Project_Details.pdf",
            "project_title": project_title,
            "category": category,
            "chunk_type": "project"
        }
        
        documents.append(Document(
            page_content=clean_content,
            metadata=metadata
        ))
    
    return documents

def seed_database(file_path: str):
    """Extracts PDF, chunks it, and uploads to vector DB."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"📄 Loading and extracting PDF: '{file_path}'...")
    
    try:
        # Extract text using pypdf
        text = extract_text_from_pdf(file_path)
    except Exception as e:
        print(f"❌ Failed to read PDF: {e}")
        return

    print("✂️ Applying project-based chunking strategy...")
    documents = chunk_projects(text)
    
    if not documents:
        print("⚠️ No project chunks found. Check if the PDF text extraction is clean.")
        return

    print(f"✅ Created {len(documents)} project chunks.")
    
    # Sample Preview
    print("\n📋 Sample chunk:")
    print(f"  Title: {documents[0].metadata['project_title']}")
    print(f"  Content: {documents[0].page_content[:150]}...")
    
    print(f"\nEmbedding and uploading {len(documents)} projects to Pinecone...")
    vector_store.add_documents(documents)
    print("🚀 Successfully seeded vector database!")
    
    return documents

if __name__ == "__main__":
    filename = "Project_Details.pdf" 
    chunks = seed_database(filename)