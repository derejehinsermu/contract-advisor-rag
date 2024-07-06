import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from visualize_advance_rag_pipeline.retrieval import vector_store

llm = ChatOpenAI(model_name="gpt-4", temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"))

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
                    "Make sure they are complete questions, and that they are related to the original question."
                    "Output one question per line. Do not number the questions."
                )
            )
        )
    ]
)

def augment_multiple_query(query):
    augmented_query = prompt_template.format(question=query)
    response = llm(augmented_query)
    print(response)  # Inspect the response object
    content = response.content.split("\n")
    return content
