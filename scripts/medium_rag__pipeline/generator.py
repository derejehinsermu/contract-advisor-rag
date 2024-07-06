from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from retrieval import query_relevant_documents, get_retriever
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate

# Initialize the LLM
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.0
)

# Define the prompt template for query expansion
prompt_template = ChatPromptTemplate(
    input_variables=['question'],
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['question'],
                template=(
                    "You are a helpful expert contract advisor assistant. Provide an example answer to the given question that might be found in a document.\n"
                    "Question: {question} \n"
                    "Answer:"
                )
            )
        )
    ]
)

# Function to augment query using the prompt template
def augment_query_generated(query):
    augmented_query = prompt_template.format(question=query)
    response = llm(augmented_query)
    return response.content

def generate_response(question):
    # Augment the query
    hypothetical_answer = augment_query_generated(question)
    joint_query = f"{question} {hypothetical_answer}"
    
    # Retrieve documents using the joint query
    retrieved_documents = query_relevant_documents(joint_query)
    context = "\n\n".join([doc.page_content for doc in retrieved_documents])

    retriever = get_retriever()
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    
    return qa({"query": question, "context": context})
