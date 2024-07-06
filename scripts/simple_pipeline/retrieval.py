import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from dotenv import load_dotenv, find_dotenv

# Unset the environment variables if they are set
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]
if "MONGODB_URI" in os.environ:
    del os.environ["MONGODB_URI"]

# Load environment variables from the .env file and explicitly reload them
load_dotenv(override=True)

# Retrieve environment variables
db_uri = os.getenv("MONGODB_URI")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if OpenAI API key is loaded
if not openai_api_key:
    raise ValueError("OpenAI API key not found! Please set 'OPENAI_API_KEY' environment variable.")

# Database configuration
client = MongoClient(db_uri, server_api=ServerApi('1'))
LizzyAI_ContractsDB = client["LizzyAI_Contracts"]
lizzyai_collection = LizzyAI_ContractsDB["contracts"]

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

vectorStore = MongoDBAtlasVectorSearch(
    embedding=embeddings, 
    collection=lizzyai_collection,
    index_name="vector_index"
)

def add_documents_to_vector_store(chunks):
    vectorStore.add_documents(chunks)

def get_retriever():
    return vectorStore.as_retriever(search_type="similarity",
                                    search_kwargs={"k":5})

