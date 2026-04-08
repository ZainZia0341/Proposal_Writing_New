llm.py

from .config import LLM_PROVIDER, GOOGLE_API_KEY, GROQ_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

def get_llm():
    if LLM_PROVIDER == "groq":
        return ChatGroq(
            temperature=0.1, 
            model_name="openai/gpt-oss-120b", # "llama3-8b-8192", 
            api_key=GROQ_API_KEY
        )
    else:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", #  "gemini-3-flash-preview", # "gemini-3.1-flash-lite-preview", # "gemini-2.5-pro", 
            temperature=0.1, 
            google_api_key=GOOGLE_API_KEY
        )

llm = get_llm()