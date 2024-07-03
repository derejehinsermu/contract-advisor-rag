from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_document(documents, chunk_size=1000, chunk_overlap=300):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)
