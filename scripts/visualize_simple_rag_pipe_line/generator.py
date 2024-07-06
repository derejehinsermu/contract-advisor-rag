from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from visualize_simple_rag_pipe_line.retrieval import vector_store

llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)

def generate_response(query):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
    return qa.invoke(query)
