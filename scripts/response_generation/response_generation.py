from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#import openai
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
#from query_expansion import expand_query_multiple, expand_query_hypothetical
from langchain_core.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.memory import ConversationBufferMemory
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.retriever import retriever

load_dotenv()

LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = "Contract Question and Answer"

openai_key = os.getenv("OPENAI_API_KEYS")
pinecone_key = os.getenv("PINECONE_API_KEYS")

chatopenai_client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEYS"), temperature=0)


retriever = retriever()

memory = ConversationBufferMemory()
#qa = RetrievalQA.from_llm(llm=chatopenai_client, retriever = retriever,  memory = memory)
qa = RetrievalQA.from_chain_type(llm=chatopenai_client, retriever= retriever, chain_type= "stuff")



async def generate_response(query):
    """
    Generate a response to a given query using LangChain and OpenAI.

    Args:
        query (str): The query string about the contract.

    Returns:
        str: Response generated by the system.
    """
    # prompt_template = """
    # You are an AI expert specialized in contract analysis and advisory services.
    # Your goal is to assist users by providing precise and helpful answers based on a contract document.
    # The document in question is an Advisory Services Agreement.
    # Given the user query, generate a well-structured and informative response by extracting relevant information from the document.
    # Ensure the response is concise, accurate, and directly addresses the query.

    # Context: {context}

    # Query: {Query}        
    # """

    response = await qa.arun({"query": query})
    return response

if __name__ == "__main__":
    #query = "Who are the parties to the Agreement and what are the their defined names?"
    query1 = "Who are the parties to the Agreement and what are their defined names?"

    # hypthotetical_asnwer = expand_query_hypothetical(query1)

    relevant_docs = retriever.get_relevant_documents(query1)

    # joint_query = f"{query1} {hypthotetical_asnwer}"
    # print(joint_query)
    response = asyncio.run(generate_response(query1, relevant_docs))

    print(response)