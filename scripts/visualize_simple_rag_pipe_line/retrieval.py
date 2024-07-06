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

def query_relevant_documents(query, vector_store, k=5):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)
