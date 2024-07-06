from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from retrieval import get_retriever

def generate_response(question):
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.0
    )

    retriever = get_retriever()
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    
    return qa({"query": question})
