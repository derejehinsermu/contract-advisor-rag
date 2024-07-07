import streamlit as st
import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file
_ = load_dotenv(find_dotenv())

# Retrieve environment variables
db_uri = os.getenv("MONGODB_URI")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Database configuration
client = MongoClient(db_uri, server_api=ServerApi('1'))

# Updated database and collection names
LizzyAI_ContractsDB = client["LizzyAI_Contracts"]
lizzyai_collection = LizzyAI_ContractsDB["contracts"]

# Define a function to load documents
def load_document(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == '.docx':
        loader = Docx2txtLoader(file_path)
    elif file_extension == '.txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    return loader.load()

# Split the data into chunks
def split_data(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(data)

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.0, openai_api_key=openai_api_key)

# Initialize the VectorStore
vectorStore = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=lizzyai_collection,
    index_name="vector_index"
)

# Set up the retriever and QA chain
retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, return_source_documents=False)

def generate_response(query):
    response = qa.invoke(query)
    return response['result']

import hashlib

def hash_document_content(content):
    """Generate a hash for the document content."""
    hasher = hashlib.md5()
    hasher.update(content.encode('utf-8'))
    return hasher.hexdigest()

def document_exists(collection, doc_hash):
    """Check if a document with the given hash already exists in the collection."""
    return collection.find_one({"doc_hash": doc_hash}) is not None

def add_document_to_vector_store(content, chunks, collection):
    """Add document chunks to the vector store and store the hash in the collection."""
    doc_hash = hash_document_content(content)
    if not document_exists(collection, doc_hash):
        vectorStore.add_documents(chunks)
        collection.insert_one({"doc_hash": doc_hash})
        return True
    return False

# Model selection in sidebar
selected_model = None
with st.sidebar:
    st.header("OpenAI Configuration")
    selected_model = st.selectbox("Model", ['gpt-3.5-turbo', 'gpt-4'], index=1)
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

# Streamlit app interface
st.title("Contract Advisor RAG System")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open(os.path.join("/tmp", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and split the document
    file_path = os.path.join("/tmp", uploaded_file.name)
    data = load_document(file_path)
    content = " ".join([doc.page_content for doc in data])
    chunks = split_data(data)
    
    # Add chunks to the vector store if not already added
    if add_document_to_vector_store(content, chunks, lizzyai_collection):
        st.success(f"Document {uploaded_file.name} processed and added to vector store.")
    else:
        st.warning(f"Document {uploaded_file.name} already exists in the vector store.")

# Display chat history
for message in st.session_state.chat_history:
    if message['role'] == 'human':
        with st.chat_message("user"):
            st.markdown(message['content'])
    else:
        with st.chat_message("assistant"):
            st.markdown(message['content'])

# Get user input
question = st.chat_input("Enter your question:")

if question:
    if question is not None and question != "":
        # Add the user's question to the chat history
        st.session_state.chat_history.append({"role": "human", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        # Get response from the model
        response = generate_response(question)

        with st.chat_message("assistant"):
            st.markdown(response)

        # Add the AI's response to the chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
