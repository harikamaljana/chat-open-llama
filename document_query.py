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
    # First, check if query is constitution-related
    CONSTITUTION_KEYWORDS = [
        'constitution', 'amendment', 'article', 'bill of rights', 'founding', 'federal',
        'congress', 'president', 'supreme court', 'rights', 'powers', 'states',
        'judicial', 'legislative', 'executive', 'ratif'
    ]
    
    is_constitution_related = any(keyword in query_text.lower() for keyword in CONSTITUTION_KEYWORDS)
    if not is_constitution_related:
        return type('Response', (), {
            'response': "I can only answer questions about the U.S. Constitution. Please ask a constitution-related question.",
            'response_gen': (chunk for chunk in ["I can only answer questions about the U.S. Constitution. Please ask a constitution-related question."]),
            'source_nodes': []
        })

    # Format chat history for context
    chat_context = ""
    if chat_history:
        chat_context = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in chat_history
        ])

    CUSTOM_QUERY_TEMPLATE = (
        "You are a Constitution-only assistant. You must REFUSE to answer ANY question not directly "
        "about the U.S. Constitution and its amendments.\n\n"
        "STRICT RULES:\n"
        "1. ONLY answer questions explicitly about the U.S. Constitution\n"
        "2. If a question is not about the Constitution, respond EXACTLY with:\n"
        "   'I can only answer questions about the U.S. Constitution. Please ask a constitution-related question.'\n"
        "3. Use ONLY information from the provided context\n"
        "4. If the constitutional information isn't in the context, respond EXACTLY with:\n"
        "   'I apologize, but I don't have enough information in my knowledge base to answer this constitutional question.'\n"
        "5. NEVER use external knowledge or make assumptions\n\n"
        "Previous conversation:\n{chat_history}\n\n"
        "Context from documents:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Question: {query_str}\n"
        "Answer: "
    )
    query_prompt = PromptTemplate(
        CUSTOM_QUERY_TEMPLATE,
        chat_history=chat_context
    )
    
    # Create query engine with streaming enabled
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="tree_summarize",
        node_postprocessors=[],
        context_window=3072,
        text_qa_template=query_prompt,
        temperature=0.1,
        streaming=True  # Enable streaming
    )
    
    try:
        response = query_engine.query(query_text)
        return response
    except Exception as e:
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