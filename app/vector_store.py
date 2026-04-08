# vector_store.py

import time
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from .config import PINECONE_API_KEY, PINECONE_INDEX_NAME, GOOGLE_API_KEY

# Google's free embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",  # "models/text-embedding-004", # "gemini-embedding-001", 
    google_api_key=GOOGLE_API_KEY,
    output_dimensionality=768,
)

pc = Pinecone(api_key=PINECONE_API_KEY)

def init_pinecone():
    """Checks if index exists, creates it if not."""
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768, # Dimension for Google embedding-001
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            time.sleep(1)
        print("Index created successfully.")
    
    return PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

vector_store = init_pinecone()

def get_retriever():
    # Retrieve top 2 most relevant projects
    return vector_store.as_retriever(search_kwargs={"k": 4})

# --- UTILITY TO SEED YOUR DATABASE ---
# Run this once manually to upload your portfolio. Notice how 1 chunk = 1 full project.
# def seed_database():
#     from langchain_core.documents import Document
#     projects = [
#         Document(page_content="Project: Homeworke (www.homeworke.com)\nOverview: All-in-one home services platform...\nTech: Bubble.io, APIs...", metadata={"type": "bubble"}),
#         Document(page_content="Project: Contextufy / Aha-doc\nOverview: AI-powered file management with RAG pipeline...\nTech: AWS S3, DynamoDB, Python, LangChain...", metadata={"type": "ai_saas"}),
#         # Add the rest of your projects here...
#     ]
#     vector_store.add_documents(projects)