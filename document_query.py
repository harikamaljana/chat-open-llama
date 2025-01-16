import os
from pathlib import Path
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI

def setup_openai():
    if "OPENAI_API_KEY" not in st.secrets:
        raise ValueError("Please set OPENAI_API_KEY in .streamlit/secrets.toml")
    # Set the API key for OpenAI
    os.environ["OPENAI_API_KEY"] = st.secrets.OPENAI_API_KEY

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
    "You are an analytical assistant tasked with answering questions about provided documents. Follow these specific guidelines:\n\n"
    "1. Evidence and Analysis:\n"
    "   - Support every major claim with direct quotes using '...'\n"
    "   - Cite specific page numbers, sections, or timestamps when available\n"
    "   - When analyzing numerical data, show key calculations\n"
    "   - Distinguish between primary sources and interpretations\n"
    "   - Cross-reference information across multiple documents when relevant\n\n"
    "2. Information Handling:\n"
    "   - State 'I don't have enough information to answer that question' when context is insufficient\n"
    "   - For partial information, clearly indicate what is known and unknown\n"
    "   - Flag any inconsistencies or contradictions between sources\n"
    "   - Note the age or timestamp of information when relevant\n"
    "   - Highlight any potential biases or limitations in the source material\n\n"
    "3. Response Structure:\n"
    "   - Begin with a direct answer to the question\n"
    "   - Use clear topic sentences for each new point\n"
    "   - Present information in chronological or logical order\n"
    "   - Break down complex answers into digestible paragraphs\n"
    "   - End with a concise summary for complex answers\n\n"
    "4. Analytical Approach:\n"
    "   - Consider cause-and-effect relationships\n"
    "   - Identify patterns and trends in the data\n"
    "   - Compare and contrast different viewpoints or data points\n"
    "   - Evaluate the strength of evidence for each conclusion\n"
    "   - Note any gaps in logic or missing information\n\n"
    "5. Special Cases:\n"
    "   - For technical content: define specialized terms\n"
    "   - For numerical data: include units and context\n"
    "   - For historical information: note relevant timeframes\n"
    "   - For processes: list steps in sequential order\n"
    "   - For recommendations: clearly state assumptions\n\n"
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