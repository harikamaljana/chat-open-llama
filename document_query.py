import os
from pathlib import Path
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI

def setup_openai():
    if "OPENAI_API_KEY" not in st.secrets:
        raise ValueError("Please set OPENAI_API_KEY in .streamlit/secrets.toml")
    
    # Initialize OpenAI with API key from secrets
    llm = OpenAI(api_key=st.secrets.OPENAI_API_KEY)
    return llm

def load_and_index_documents(data_dir: str, storage_dir: str):
    # Create storage directory if it doesn't exist
    Path(storage_dir).mkdir(exist_ok=True)
    
    storage_path = Path(storage_dir) / "docstore.json"
    # Check if we have an existing index
    if storage_path.exists():
        try:
            # Load existing index
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(storage_context)
            print("Loaded existing index from storage")
        except Exception as e:
            print(f"Error loading existing index: {e}")
            print("Creating new index...")
            return create_new_index(data_dir, storage_dir)
    else:
        return create_new_index(data_dir, storage_dir)
    
    return index

def create_new_index(data_dir: str, storage_dir: str):
    # Create new index
    print("Creating new index...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    # Save index to disk
    index.storage_context.persist(persist_dir=storage_dir)
    print("Index created and saved to storage")
    
    return index

def query_documents(index, query_text: str, chat_history=None):
    # Format chat history for context
    chat_context = ""
    if chat_history:
        chat_context = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in chat_history
        ])
    
    # Create query engine with optimized settings
    query_engine = index.as_query_engine(
        similarity_top_k=5,  # Increased from 3 to get more context
        response_mode="tree_summarize",
        node_postprocessors=[],
        context_window=4096,  # Increased context window
        text_qa_template=query_prompt,
        temperature=0.7,  # Increased temperature for more creative responses
        streaming=True,
        verbose=True  # Add this to see more details about the query process
    )
    
    try:
        # Add system message to guide response format
        query_text_with_instruction = (
            "Please provide a detailed answer with examples and evidence from the documents. "
            "Break down complex information into clear sections. "
            "QUERY: " + query_text
        )
        
        response = query_engine.query(query_text_with_instruction)
        return response
    except Exception as e:
        print(f"Query error: {e}")  # Add error logging
        return "Cannot answer based on the available information."

# def main():
#     # Setup OpenAI credentials
#     setup_openai()
    
#     # Specify directories
#     data_dir = "data"
#     storage_dir = "storage"
    
#     # Create directories if they don't exist
#     Path(data_dir).mkdir(exist_ok=True)
    
#     # Load and index documents
#     index = load_and_index_documents(data_dir, storage_dir)
    
#     # Example query
#     query = "What are the Partner Tools and Accelerators that can be utilized for RISE with SAP Engagements?"
#     response = query_documents(index, query)
    
#     print(f"\nQuery: {query}")
#     print(f"Response: {response}")

# if __name__ == "__main__":
#     main() 