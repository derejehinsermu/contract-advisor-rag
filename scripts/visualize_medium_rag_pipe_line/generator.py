import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate

llm = ChatOpenAI(model_name="gpt-4", temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"))

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

def augment_query_generated(query):
    augmented_query = prompt_template.format(question=query)
    response = llm(augmented_query)
    return response
