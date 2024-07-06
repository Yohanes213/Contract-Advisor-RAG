from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEYS")
pinecone_key = os.getenv("PINECONE_API_KEYS")


def retriever():
    """
    Initializes and returns a retriever using Pinecone and OpenAI embeddings.

    Returns:
    retriever: A configured retriever for searching through the Pinecone vector store.
    """
    pc = PineconeClient(pinecone_key)
    index_name = "lawquestionandanswer"
    embed_model = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=openai_key
    )

    vectordb = PineconeVectorStore(
        embedding=embed_model,
        pinecone_api_key=pinecone_key,
        index_name=index_name,
        namespace="Robinson",
    )

    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.25}
    )

    return retriever
