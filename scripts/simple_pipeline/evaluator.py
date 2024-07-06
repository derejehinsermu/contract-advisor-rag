import pandas as pd
import json
from ragas.langchain.evalchain import RagasEvaluatorChain
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
)

def load_evaluation_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def evaluate_responses(responses):
    eval_chains = {
        m.name: RagasEvaluatorChain(metric=m)
        for m in [faithfulness, answer_relevancy, context_relevancy, context_recall]
    }

    evaluation_results = []
    for response in responses:
        result = {
            'questions': response['query'],
            'contexts': "\n\n".join([doc.page_content for doc in response['source_documents']]),
            'answer': response['result'],
            'ground_truth': response['ground_truths'][0]
        }
        for name, eval_chain in eval_chains.items():
            score_name = f"{name}_score"
            score = eval_chain(response)[score_name]
            result[score_name] = score
        evaluation_results.append(result)
    
    return evaluation_results

def print_evaluation_results(results):
    df_results = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(df_results)
