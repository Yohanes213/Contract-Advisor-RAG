from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore
import os
#import openai
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEYS")
pinecone_key = os.getenv("PINECONE_API_KEYS")

chatopenai_client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEYS"), temperature=0)

def retriever():

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