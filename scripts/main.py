#from scripts.query_expansion import expand_query_hypothetical, expand_query_multiple
#from scripts.rank import coherererank
from scripts.response_generation.response_generation import generate_response
from evaluation.evaluation import evaluate_metrics
from evaluation.evaluation_result import visualize_result
from scripts.reranking.rank import coherererank
from query_expansion import expand_query_multiple

import asyncio


async def run(query):
    
    response = await generate_response(query)

    return response


def run_expand_query_multiple(original_query):
    queries = expand_query_multiple(original_query)
    return queries

def run_cohere_rerank():
    ranked_retrived_context = coherererank()
    return ranked_retrived_context

def run_evaluate():

    result = evaluate_metrics()

    return result

def run_visualize_result(result, name):

    visualize_result(result, name)


if __name__ == "__main__":
    result = run_evaluate()
    print(result)

    visualize_result(result, 'result_html/3. rag_evaluation for RetrivalQA(RetrievalQA.from_chain_type).html', 'Retrival Augmented Generation (RetrievalQA.from_chain_type)- Evaluation')
    #query = 'What is the termination notice?'
    #result = asyncio.run(run(query))

    print(result)
