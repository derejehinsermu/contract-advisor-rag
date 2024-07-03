import os
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

def initialize_retriever(chunks=None):
    embeddings = OpenAIEmbeddings()
    Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)  # Initialize Pinecone
    index_name = "vector-store"  # Define Pinecone index name
    if chunks:
        docsearch = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=index_name
        )
    else:
        docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    return docsearch
