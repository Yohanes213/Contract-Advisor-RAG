from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient
from pathlib import Path
import os
from document_extractor import extract_text_from_pdf
import asyncio
from logger import logger

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEYS')
pinecone_key = os.getenv('PINECONE_API_KEYS')

async def embed_text(query, model='text-embedding-ada-002'):
    """
    Embed a query using the specified OpenAI embedding model.

    Args:
    - query (str): The query to embed.
    - model (str): Name of the OpenAI embedding model (default: 'text-embedding-ada-002').

    Returns:
    - list: List of embeddings representing the query.
    """
    try:
        embed_model = OpenAIEmbeddings(model=model, openai_api_key=openai_key)
        document_embeddings = embed_model.embed_query(query)
        return document_embeddings
    except Exception as e:
        logger.error(f"Error embedding query '{query}': {str(e)}")
        return []

async def querying(embedded_query):
    """
    Query a Pinecone index with the embedded query.

    Args:
    - embedded_query (list): List of embeddings representing the query.

    Returns:
    - dict: Retrieved documents matching the query.
    """
    try:
        pc = PineconeClient(pinecone_key)
        index_name = 'lawquestionandanswer'
        index = pc.Index(index_name)
        
        retrieved = index.query(
            namespace="Robinson",
            vector=embedded_query,
            top_k=2,
            include_metadata=True,
            include_vectors=False  # Only fetch metadata, not vectors
        )
        return retrieved
    except Exception as e:
        logger.error(f"Error querying with embedded query: {str(e)}")
        return {}

async def main():
    query = "Can the subject pay the Advisor?"
    embedded_text = await embed_text(query)
    result = await querying(embedded_text)

    # Extract and print only document text from retrieved results
    text_results = [match['metadata']['text'] for match in result['matches']]
    print(f"Found {len(text_results)} relevant documents")
    print(text_results)

if __name__ == "__main__":
    asyncio.run(main())