import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def setup_openai():
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
    "You are an analytical assistant tasked with answering questions about provided documents. Follow these guidelines:\n\n"
    "1. Base all responses on the provided context and chat history:\n"
    "   - Support claims with direct quotes using '...' or specific references\n"
    "   - Clearly indicate if information is ambiguous or incomplete\n"
    "   - State 'I don't have enough information to answer that question' when context is insufficient\n"
    "   - Note any uncertainties or limitations in your analysis\n\n"
    "2. Structure your responses:\n"
    "   - Be concise and precise\n"
    "   - Lead with clear topic sentences\n"
    "   - Format evidence consistently\n"
    "   - Avoid speculation beyond provided information\n\n"
    "3. When analyzing:\n"
    "   - Consider temporal context and relationships\n"
    "   - Note any contradictions or inconsistencies\n"
    "   - Distinguish between explicit statements and implications\n"
    "   - Identify connections between different parts of the context\n\n"
    "Previous conversation:\n"
    "{chat_history}\n\n"
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
    
    # Create query engine with improved settings
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="tree_summarize",
        node_postprocessors=[],
        context_window=3072,
        text_qa_template=query_prompt,
        temperature=0.1
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