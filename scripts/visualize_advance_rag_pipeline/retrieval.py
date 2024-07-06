import os
import numpy as np
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv, find_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch

load_dotenv(find_dotenv())  # Load environment variables from the .env file

# Retrieve environment variables
db_uri = os.getenv("MONGODB_URI")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Database configuration
client = MongoClient(db_uri, server_api=ServerApi('1'))
LizzyAI_ContractsDB = client["LizzyAI_Contracts"]
lizzyai_collection = LizzyAI_ContractsDB["contracts"]

embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings_model,
    collection=lizzyai_collection,
    index_name="vector_index"
)

def load_embeddings_from_db(collection):
    embeddings = []
    documents = []
    for doc in collection.find({}, {"embedding": 1, "text": 1, "_id": 0}):
        embeddings.append(doc["embedding"])
        documents.append(doc["text"])
    return np.array(embeddings), documents

def query_relevant_documents(queries, vector_store, k=5):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    all_documents = []
    all_embeddings = []
    for query in queries:
        results = retriever.get_relevant_documents(query)
        documents = [doc.page_content for doc in results]
        embeddings = [doc.metadata['embedding'] for doc in results]
        all_documents.append(documents)
        all_embeddings.append(embeddings)
    return all_documents, all_embeddings
