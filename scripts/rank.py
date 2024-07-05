from dotenv import load_dotenv
import os
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from query_expansion import expand_query_hypothetical, expand_query_multiple
from response_generation import generate_response
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
import asyncio
from retriever import retriever

load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEYS")
os.environ["COHERE_API_KEY"] = cohere_api_key
openai_key = os.getenv("OPENAI_API_KEYS")
pinecone_key = os.getenv("PINECONE_API_KEYS")


retriever = retriever()

def coherererank():
    """
    Creates and configures the retrieval pipeline with ContextualCompressionRetriever
    for efficient retrieval and ranking.

    Returns:
        ContextualCompressionRetriever: The configured retriever instance.
    """

    llm = Cohere(temperature=0) 
    compressor = CohereRerank()  

    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    return compressed_retriever

if __name__ == "__main__":
    compressed_retriever = coherererank()

    original_query = "Can the Agreement or any of its obligations be assigned?"
    joint_query = expand_query_multiple(original_query)

    list_queries = joint_query.strip().split('\n')
    list_queries.append(original_query) 

    
    all_docs = []
    for query1 in list_queries:
        docs = compressed_retriever.invoke(query1)
        all_docs.extend(docs)

    ranked_docs = sorted(all_docs, key=lambda doc: vars(doc)['metadata']['relevance_score'], reverse=True)  # Example ranking by score

    query =vars(ranked_docs[0])['page_content']
    
    response = asyncio.run(generate_response(original_query,query))

    print(response)