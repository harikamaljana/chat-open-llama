import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def setup_openai():
    # Make sure you have set OPENAI_API_KEY in your environment variables
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")

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

def query_documents(index, query_text: str):
    # Define custom query prompt
    CUSTOM_QUERY_TEMPLATE = (
    "You are a helpful assistant that answers questions based on the provided documents. "
    "Only answer questions that can be answered using the provided document content. "
    "Do not mention the documents in your response. "
    "If a question cannot be answered using the documents, politely explain that you can only "
    "answer questions related my knowledge base.\n\n"
    "Context information is below:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given this context, please answer the question: {query_str}\n"
    )
    
    query_prompt = PromptTemplate(CUSTOM_QUERY_TEMPLATE)
    
    # Create query engine with strict mode
    query_engine = index.as_query_engine(
        similarity_top_k=1,
        response_mode="compact",
        node_postprocessors=[],
        context_window=2048,
        text_qa_template=query_prompt,
        temperature=0.0
    )
    
    try:
        response = query_engine.query(query_text)
        
        if (not response.response or 
            response.response.strip() == "" or 
            len(response.source_nodes) == 0):
            return "Cannot answer based on the available information."
            
        return response
    except Exception as e:
        return "Cannot answer based on the available information."

def main():
    # Setup OpenAI credentials
    setup_openai()
    
    # Specify directories
    data_dir = "data"
    storage_dir = "storage"
    
    # Create directories if they don't exist
    Path(data_dir).mkdir(exist_ok=True)
    
    # Load and index documents
    index = load_and_index_documents(data_dir, storage_dir)
    
    # Example query
    query = "What are the Partner Tools and Accelerators that can be utilized for RISE with SAP Engagements?"
    response = query_documents(index, query)
    
    print(f"\nQuery: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main() 