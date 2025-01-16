import streamlit as st
from document_query import setup_openai, load_and_index_documents, query_documents
from pathlib import Path

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """👋 Welcome to Constitution Chat! 
            How can I help you learn about the Constitution today?"""
        }
    ]

# Initialize session state for document index
if "index" not in st.session_state:
    st.set_page_config(page_title="Constitution Chat", layout="wide")
    setup_openai()
    
    # Specify directories
    data_dir = "data"
    storage_dir = "storage"
    
    # Create directories if they don't exist
    Path(data_dir).mkdir(exist_ok=True)
    
    # Load and index documents
    st.session_state.index = load_and_index_documents(data_dir, storage_dir)

def clear_chat():
    st.session_state.messages = []

def main():
    # Create a container for the header
    header = st.container()
    with header:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.title("Constitution Chat")
        with col2:
            st.button("Clear Chat", on_click=clear_chat, type="primary")
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("source"):
                with st.expander("Source"):
                    st.markdown(message["source"])
    
    # Accept user input
    if prompt := st.chat_input("Ask a question about the constitution"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate assistant response using document_query backend
        with st.chat_message("assistant"):
            chat_history = st.session_state.messages[-4:] if len(st.session_state.messages) > 0 else None
            
            # Create empty placeholder for streaming response
            response_placeholder = st.empty()
            full_response = ""
            
            # Get streaming response
            response = query_documents(
                st.session_state.index, 
                prompt,
                chat_history=chat_history
            )
            
            # Stream the response
            for chunk in response.response_gen:
                if chunk:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
            
            # Display final response without cursor
            response_placeholder.markdown(full_response)
            
            # Show source if available
            if hasattr(response, 'source_nodes') and response.source_nodes:
                source_text = response.source_nodes[0].node.text
                with st.expander("Source"):
                    st.markdown(source_text)
        
        # Add assistant response to chat history with source
        source = response.source_nodes[0].node.text if hasattr(response, 'source_nodes') and response.source_nodes else None
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "source": source
        })

if __name__ == "__main__":
    main() 