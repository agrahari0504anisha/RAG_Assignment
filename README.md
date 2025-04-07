# Customer Support RAG Chatbot

A Streamlit-based chatbot that uses Retrieval-Augmented Generation (RAG) to answer customer support questions based on your documentation.

## Overview

This application creates a conversational interface that allows users to ask questions about your product or service. It uses document embeddings and vector search to find relevant information from your knowledge base, then generates accurate answers using Azure OpenAI's language models.

## Features

- **Multi-format Document Support**: Processes PDF, DOCX, TXT, and other file formats
- **Web Content Integration**: Can retrieve information from specified websites
- **Vector Search**: Uses FAISS for efficient similarity search
- **Conversational UI**: Clean chat interface powered by Streamlit
- **Context-aware Responses**: Generates answers based only on available documentation
- **Fallback Handling**: Responds with "I don't know" when information isn't available

## Prerequisites

- Python 3.8+
- Azure OpenAI API access
- A directory containing your knowledge base documents

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd customer-support-rag-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with the following variables:
   ```
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_API_VERSION=your_api_version
   DEPLOYMENT_NAME=your_deployment_name
   EMBEDDING_MODEL=your_embedding_model
   DOCUMENT_PATH=path/to/your/documents
   WEBSITE_URLS=https://example.com,https://another-example.com
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run rag-chatbot.py
   ```

2. Open your browser and navigate to the provided URL (typically http://localhost:8501)

3. Ask questions in the chat interface about your product or documentation

## How It Works

1. **Document Processing**: The application loads documents from the specified directory and splits them into manageable chunks.

2. **Embedding Generation**: Each chunk is converted into a vector embedding using Azure OpenAI's embedding model.

3. **Vector Storage**: Embeddings are stored in a FAISS vector database for efficient similarity search.

4. **Query Processing**: When a user asks a question, it's converted to an embedding and used to find the most relevant document chunks.

5. **Response Generation**: The relevant chunks are sent to Azure OpenAI's model along with the question to generate a contextually appropriate answer.

## Customization

### Modifying the Prompt Template

You can customize the prompt template in the `create_qa_chain` function to change how the model formats its responses or handles certain types of questions.

### Adjusting Chunk Size

If your answers are missing context or are too fragmented, you can adjust the `chunk_size` and `chunk_overlap` parameters in the `RecursiveCharacterTextSplitter`.

### Adding More Document Types

Support for additional document types can be added by implementing appropriate loaders in the `load_and_process_documents` function.

## Troubleshooting

- **No documents found**: Check that your `DOCUMENT_PATH` environment variable points to a directory containing supported document types.
- **API errors**: Verify your Azure OpenAI API credentials and make sure your subscription is active.
- **Memory issues**: If processing large document collections, you may need to increase available memory or reduce the number of documents.

## Requirements

```
streamlit
langchain
langchain-community
langchain-openai
faiss-cpu
python-dotenv
pypdf
docx2txt
unstructured
requests
bs4
```