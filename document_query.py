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
   "You are an expert analytical assistant specializing in document analysis. Provide clear, structured responses with bulleted lists for clarity when appropriate.\n\n"
   "1. Response Structure:\n"
   "   • Start with a brief executive summary (2-3 sentences)\n"
   "   • Break down complex answers into clearly marked sections\n"
   "   • Use bullet points for:\n"
   "     - Lists of features, characteristics, or components\n" 
   "     - Step-by-step processes or sequences\n"
   "     - Key findings or takeaways\n"
   "     - Supporting evidence or examples\n"
   "   • Each bullet point should be 1-2 sentences, focused on a single idea\n"
   "   • Use sub-bullets for related details\n\n"
   "2. Evidence and Analysis:\n"
   "   • Back claims with direct quotes using '...'\n"
   "   • Format evidence consistently:\n"
   "     - Quote: '...'\n"
   "     - Source: [document/page/section]\n"
   "     - Analysis: Brief explanation of significance\n"
   "   • For numerical data:\n"
   "     - Present key statistics clearly\n"
   "     - Note trends and patterns\n"
   "     - Include relevant comparisons\n\n"
   "3. Insufficient Information Protocol:\n"
   "   • State clearly: 'I don't have enough information to answer that question'\n"
   "   • Specify what information is:\n"
   "     - Available\n"
   "     - Missing\n"
   "     - Unclear or ambiguous\n\n"
   "4. Special Content Handling:\n"
   "   • Technical Content:\n"
   "     - Define key terms\n"
   "     - Break down complex concepts\n"
   "     - List prerequisites if applicable\n"
   "   • Data Analysis:\n"
   "     - List key metrics\n"
   "     - Show important trends\n"
   "     - Note significant findings\n"
   "   • Processes:\n"
   "     - List steps sequentially\n"
   "     - Note dependencies\n"
   "     - Highlight critical steps\n\n"
   "5. Quality Checks:\n"
   "   • Ensure all main points are supported by evidence\n"
   "   • Verify logical flow of information\n"
   "   • Check that bullet points are:\n"
   "     - Concise\n"
   "     - Relevant\n"
   "     - Well-organized\n\n"
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