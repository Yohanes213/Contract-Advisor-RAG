#from scripts.query_expansion import expand_query_hypothetical, expand_query_multiple
#from scripts.rank import coherererank
from scripts.response_generation.response_generation import generate_response
from scripts.evaluation.evaluation import evaluate_metrics
from scripts.evaluation.evaluation_result import visualize_result
from scripts.reranking.rank import coherererank
from scripts.query_expansion import expand_query_multiple

import asyncio


async def run(query):
    """
    Asynchronously generates a response for a given query using response generation scripts.

    Parameters:
    query (str): The query for which the response is generated.

    Returns:
    str: The generated response.
    """
    response = await generate_response(query)

    return response


def run_expand_query_multiple(original_query):
    """
    Expands a given query into multiple related queries using query expansion scripts.

    Parameters:
    original_query (str): The original query to expand.

    Returns:
    str: Multiple related queries.
    """
    queries = expand_query_multiple(original_query)
    return queries


def run_cohere_rerank():
    """
    Runs the coherence reranking process using rank scripts.

    Returns:
    list: Ranked and retrieved contexts.
    """
    ranked_retrived_context = coherererank()
    return ranked_retrived_context


def run_evaluate():
    """
    Evaluates metrics for generated responses using evaluation scripts.

    Returns:
    dict: Evaluation metrics.
    """
    result = evaluate_metrics()
    return result


def run_visualize_result(result, name):
    """
    Visualizes evaluation results using visualization scripts.

    Parameters:
    result (dict): Evaluation results to visualize.
    name (str): Name of the output HTML file where the visualization will be saved.

    Returns:
    None: Saves the visualization as an HTML file.
    """
    visualize_result(result, name)


if __name__ == "__main__":
    result = run_evaluate()
    print(result)

    visualize_result(result, 'result_html/3. rag_evaluation for RetrivalQA(RetrievalQA.from_chain_type).html', 'Retrival Augmented Generation (RetrievalQA.from_chain_type)- Evaluation')
    #query = 'What is the termination notice?'
    #result = asyncio.run(run(query))

    print(result)
