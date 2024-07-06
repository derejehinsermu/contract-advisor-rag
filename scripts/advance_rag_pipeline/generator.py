from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from retrieval import query_relevant_documents, get_retriever
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from sentence_transformers import CrossEncoder

# Initialize the LLM
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.0
)

# Define the prompt template for multiple query expansion
prompt_template = ChatPromptTemplate(
    input_variables=['question'],
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['question'],
                template=(
                    "You are a helpful expert contract advisor research assistant. Your users are asking questions about contracts. "
                    "Suggest up to five additional related questions to help them find the information they need, for the provided question. "
                    "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic. "
                    "Make sure they are complete questions, and that they are related to the original question. "
                    "Output one question per line. Do not number the questions."
                )
            )
        )
    ]
)

# Function to augment query with multiple related questions
def augment_multiple_query(query):
    augmented_query = prompt_template.format(question=query)
    response = llm(augmented_query)
    content = response.content.split('\n')
    questions = [line.strip() for line in content if line.strip()]
    return questions

def generate_response(question):
    # Augment the query with multiple related questions
    generated_queries = augment_multiple_query(question)
    queries = [question] + generated_queries
    
    # Retrieve documents using the expanded queries
    all_retrieved_documents, retriever = query_relevant_documents(queries)
    
    # Deduplicate the retrieved documents
    unique_documents = {doc.page_content: doc for doc in all_retrieved_documents}.values()

    # Use CrossEncoder for re-ranking
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[question, doc.page_content] for doc in unique_documents]
    scores = cross_encoder.predict(pairs)

    # Print the retrieved documents with scores
    ranked_documents = sorted(zip(scores, unique_documents), reverse=True, key=lambda x: x[0])

    # Select top 5 documents
    top_5_documents = [doc.page_content for score, doc in ranked_documents[:5]]

    # Define RetrievalQA
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    # Query the LLM with the original question and the context from top 5 documents
    response = qa.invoke({"query": question, "context": "\n\n".join(top_5_documents)})

    return response
