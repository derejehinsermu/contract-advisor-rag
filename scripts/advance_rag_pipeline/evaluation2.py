from datasets import Dataset
from advance_rag_pipeline.generator import generate_response,get_retriever

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# Function to generate responses
def generate_responses(evaluation_data):
    responses = []
    for item in evaluation_data:
        question = item['question']
        ground_truth = item['answer']

        # Generate a response (replace this with your actual response generation function)
        response = generate_response(question)
        
        # Store the response and ground truth
        responses.append({
            'query': question,
            'result': response['result'],
            'source_documents': response['source_documents'],
            'ground_truths': [ground_truth]
        })
    return responses

# Function to extract fields and prepare the dataset
def prepare_dataset(responses):
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for response in responses:
        questions.append(response['query'])
        answers.append(response['result'])
        # Extract the page content from the source documents
        extracted_contexts = [doc.page_content for doc in response['source_documents']]
        contexts.append(extracted_contexts)
        ground_truths.append(response['ground_truths'])  # Append ground truth from response

    # Prepare the dataset dictionary
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths  # Ensure the key name matches expected column name
    }

    # Convert dict to dataset
    dataset = Dataset.from_dict(data)
    return dataset

# Function to evaluate the dataset
def evaluate_dataset(dataset):
    result = evaluate(
        dataset=dataset, 
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    # Convert results to DataFrame and print
    df = result.to_pandas()
    return df
