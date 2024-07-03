from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import openai
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from langchain import hub
from langchain_openai import ChatOpenAI
import os
import asyncio
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore


load_dotenv()

LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = "Contract Question and Answer"
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEYS")
pinecone_key = os.getenv("PINECONE_API_KEYS")

openai_client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEYS"), temperature=0)


def retrive():
    """
    Initialize Pinecone client and setup vector store retriever.

    Returns:
        Pinecone retriever object configured for contract question and answer retrieval.
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

    retriver = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.25}
    )

    return retriver


async def generate_response(query):
    """
    Generate a response to a given query using LangChain and OpenAI.

    Args:
        query (str): The query string for which response is generated.

    Returns:
        str: Response generated by the system.
    """
    retriver = retrive()

    prompt = hub.pull("rlm/rag-prompt")

    chain = (
        {"context": retriver, "question": RunnablePassthrough()}
        | prompt
        | openai_client
        | StrOutputParser()
    )

    response = chain.invoke(query)
    return response


if __name__ == "__main__":
    query = "Who are the parties to the Agreement and what are the their defined names?"
    print(asyncio.run(generate_response(query)))
