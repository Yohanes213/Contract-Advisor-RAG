from dotenv import load_dotenv
import os
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from query_expansion import expand_query_hypothetical, expand_query_multiple
from response_generation import retrive, generate_response
import asyncio

load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEYS")
os.environ["COHERE_API_KEY"] = cohere_api_key

def coherererank():
    """
    Creates and configures the retrieval pipeline with ContextualCompressionRetriever
    for efficient retrieval and ranking.

    Returns:
        ContextualCompressionRetriever: The configured retriever instance.
    """

    llm = Cohere(temperature=0)  # Use Cohere for reranking
    compressor = CohereRerank()  # Use Cohere for compression
    retriever = retrive()  # Replace with your preferred retrieval function

    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    return compressed_retriever

if __name__ == "__main__":
    compressed_retriever = coherererank()

    query = "Can the Agreement or any of its obligations be assigned?"
    joint_query = expand_query_multiple(query)

    list_queries = joint_query.strip().split('\n')
    list_queries.append(query)  # Include the original query

    # Retrieve documents for all expanded and original queries
    all_docs = []
    for query1 in list_queries:
        docs = compressed_retriever.invoke(query1)
        all_docs.extend(docs)

    #print(all_docs)
    # Rank all retrieved documents using a suitable ranking algorithm (replace with your choice)
    ranked_docs = sorted(all_docs, key=lambda doc: vars(doc)['metadata']['relevance_score'], reverse=True)  # Example ranking by score

    query = ''
    # Print the top 3 ranked documents
    for doc in ranked_docs[:3]:
        print(doc)
        print("\n")  # Add newline for better formatting

        query+=vars(doc)['page_content']
        query+= "\n"

    
response = asyncio.run(generate_response(query))

print(response)