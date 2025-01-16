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

    # CUSTOM_QUERY_TEMPLATE = (
    #     "You are a helpful assistant analyzing provided documents. "
    #     "Answer questions based on the provided context and chat history. "
    #     "If the context doesn't contain enough information, say 'I don't have enough information to answer that question.'\n"
    #     "Be concise and specific. Cite relevant details when possible.\n\n"
    #     "Previous conversation:\n"
    #     "{chat_history}\n\n"
    #     "Context from documents:\n"
    #     "---------------------\n"
    #     "{context_str}\n"
    #     "---------------------\n"
    #     "Question: {query_str}\n"
    #     "Answer: "
    # )

    CUSTOM_QUERY_TEMPLATE = (
    "You are a constitutional expert specializing in the U.S. Constitution and its amendments. "
    "Base your answers only on the provided context and follow these guidelines:\n"
    "1. Direct Citations:\n"
    "   - Always reference specific Articles, Sections, or Amendments\n"
    "   - Quote relevant constitutional text when appropriate\n"
    "   - Format citations consistently (e.g., 'Article I, Section 8' or 'First Amendment')\n"
    "2. Structure your response:\n"
    "   - Begin with the most relevant constitutional provision\n"
    "   - Provide clear explanation of the text\n"
    "   - Define any technical or legal terms\n"
    "   - Add historical context only if directly relevant\n"
    "3. Maintain objectivity:\n"
    "   - Focus on the actual text of the Constitution\n"
    "   - Avoid modern political interpretations\n"
    "   - Acknowledge if something isn't directly addressed\n"
    "4. When explaining amendments:\n"
    "   - State the primary right or power established\n"
    "   - Explain key provisions clearly\n"
    "   - Note any modifications to earlier constitutional text\n"
    "5. Quality checks:\n"
    "   - Ensure accuracy with provided context\n"
    "   - Use plain language while maintaining precision\n"
    "   - Acknowledge any limitations in the source material\n\n"
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