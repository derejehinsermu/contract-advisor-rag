from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def create_rag_chain(retriever, llm):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever.as_retriever(),
        return_source_documents=True
    )

def get_answer(retriever, query):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    qa_chain = create_rag_chain(retriever, llm)
    return qa_chain.invoke(query)
