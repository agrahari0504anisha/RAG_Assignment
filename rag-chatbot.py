import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import WebBaseLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#load env
load_dotenv()

@st.cache_resource
def load_and_process_documents(docs_dir=os.environ.get("DOCUMENT_PATH")):
    """Load and process documents from multiple file formats (PDF, DOCX, TXT) in the specified directory"""
    logger.info(f"Loading documents from {docs_dir}")
    
    # Initialize empty list to store all documents
    all_documents = []
    
    # Load PDF documents
    try:
        pdf_loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        pdf_documents = pdf_loader.load()
        all_documents.extend(pdf_documents)
        logger.info(f"Loaded {len(pdf_documents)} PDF documents")
    except Exception as e:
        logger.warning(f"Error loading PDF documents: {e}")
    
    # Load DOCX documents
    try:
        docx_loader = DirectoryLoader(docs_dir, glob="**/*.docx", loader_cls=Docx2txtLoader)
        docx_documents = docx_loader.load()
        all_documents.extend(docx_documents)
        logger.info(f"Loaded {len(docx_documents)} DOCX documents")
    except Exception as e:
        logger.warning(f"Error loading DOCX documents: {e}")
        
    website_urls = os.environ.get("WEBSITE_URLS","")
        
    if website_urls != "":
        try:            
            if isinstance(website_urls, str):
                website_urls = [website_urls]
                
            web_loader = WebBaseLoader(website_urls)
            web_documents = web_loader.load()
            all_documents.extend(web_documents)
            logger.info(f"Loaded {len(web_documents)} web pages")
        except Exception as e:
            logger.warning(f"Error loading web content: {e}")
    
    # Load text documents
    try:
        text_loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)
        text_documents = text_loader.load()
        all_documents.extend(text_documents)
        logger.info(f"Loaded {len(text_documents)} text documents")
    except Exception as e:
        logger.warning(f"Error loading text documents: {e}")
    
    # Fallback for other file types using UnstructuredFileLoader
    try:
        other_loader = DirectoryLoader(docs_dir, glob="**/*.*", 
                                      exclude=["**/*.pdf", "**/*.docx", "**/*.txt"],
                                      loader_cls=UnstructuredFileLoader)
        other_documents = other_loader.load()
        all_documents.extend(other_documents)
        logger.info(f"Loaded {len(other_documents)} documents of other types")
    except Exception as e:
        logger.warning(f"Error loading other document types: {e}")
    
    if not all_documents:
        raise ValueError(f"No documents found in {docs_dir}")
    
    logger.info(f"Total documents loaded: {len(all_documents)}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)
    
    logger.info(f"Split into {len(chunks)} chunks")
    
    # Create embeddings and vector store
    embeddings = AzureOpenAIEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL")
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    logger.info("Vector store created successfully")
    
    return vectorstore

def create_qa_chain(vectorstore):
    """Create a question-answering chain with the custom prompt"""
    
    # Custom prompt that instructs the model to say "I don't know" when unsure
    custom_prompt_template = """
    You are a customer support assistant that only answers questions based on the provided context.
    
    Context:
    {context}
    
    Question: {question}
    
    Instructions:
    1. Answer the question based ONLY on the provided context.
    2. If the information to answer the question is not in the context, respond with "I don't know".
    3. Be concise and direct in your answers.
    4. Do not make up information.
    
    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=custom_prompt_template, 
        input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=AzureChatOpenAI(
            model="gpt-4",
            azure_deployment=os.environ.get("DEPLOYMENT_NAME"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        ),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain

def main():
    # App title and description
    st.title("Customer Support RAG Chatbot")
    st.markdown("""
    This chatbot can answer questions based on the customer support documentation. 
    Ask a question, and I'll try to help you!
    """)
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    user_query = st.chat_input("Ask a question about our product...")
    
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Display assistant response with a spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Load the knowledge base
                    vectorstore = load_and_process_documents()
                    
                    # Create QA chain
                    qa_chain = create_qa_chain(vectorstore)
                    
                    # Get response
                    response = qa_chain({"query": user_query})
                    answer = response["result"]
                    
                    # Display response
                    st.markdown(answer)
                    
                    # Add response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    st.error(error_message)
                    logger.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()
