#from scripts.query_expansion import expand_query_hypothetical, expand_query_multiple
#from scripts.rank import coherererank
from response_generation import generate_response
from evaluation import evaluate_metrics
from evaluation_result import visualize_result

import asyncio


async def run(query):
    
    response = await generate_response(query)

    return response

def run_evaluate():

    result = evaluate_metrics()

    return result

def run_visualize_result(result, name):

    visualize_result(result, name)


if __name__ == "__main__":
    result = run_evaluate()
    print(result)

    visualize_result(result, '../1. rag_evaluation for RetrivalQA(using RecursiveCharacterTextSplitter).html')
    #query = 'What is the termination notice?'
    #result = asyncio.run(run(query))

    print(result)
