import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

def split_document(data, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)
