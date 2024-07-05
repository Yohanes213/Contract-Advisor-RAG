from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient
from dotenv import load_dotenv
from pathlib import Path
import os
from ..document_extraction.document_extractor import extract_text_from_pdf
from scripts.utils.logger import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
openapi_key = os.getenv("OPENAI_API_KEYS")
pinecone_key = os.getenv("PINECONE_API_KEYS")


def chunk_text(text, max_tokens=300):
    """
    Chunk input text into segments of approximately max_tokens tokens.

    Args:
    - text (str): The input text to be chunked.
    - max_tokens (int): Maximum number of tokens per chunk (default: 300).

    Returns:
    - list: List of chunked text segments.
    """
    chunks = []
    current_chunk = ""
    tokens_count = 0
    sentences = text.split(". ")

    for sentence in sentences:
        if tokens_count + len(sentence.split()) > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = ""
            tokens_count = 0
        current_chunk += sentence + ". "
        tokens_count += len(sentence.split())

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=100
)



def embed_text(chunked_text, model="text-embedding-ada-002"):
    """
    Embed chunked text segments using OpenAI's text embedding model.

    Args:
    - chunked_text (list): List of text segments to embed.
    - model (str): Name of the OpenAI embedding model (default: 'text-embedding-ada-002').

    Returns:
    - list: List of embedded document representations.
    """
    try:
        embed_model = OpenAIEmbeddings(model=model, openai_api_key=openapi_key)
        document_embeddings = embed_model.embed_documents(chunked_text)
        return document_embeddings
    except Exception as e:
        logger.error(f"Error embedding text: {str(e)}")
        return []


def vectorize(chunked_text, document, namespace):
    """
    Vectorize embedded text chunks and upsert into a Pinecone index.

    Args:
    - chunked_text (list): List of text chunks.
    - document (list): List of document embeddings corresponding to chunked_text.
    - namespace (str): Namespace for Pinecone index.

    Returns:
    - None
    """
    try:
        upsert_data = [
            (str(i), embedding, {"text": chunked_text[i]})
            for i, embedding in enumerate(document)
        ]

        pc = PineconeClient(pinecone_key)
        index_name = "lawquestionandanswer"
        index = pc.Index(index_name)

        index.upsert(upsert_data, namespace=namespace)
    except Exception as e:
        logger.error(f"Error vectorizing text: {str(e)}")


if __name__ == "__main__":
    result = extract_text_from_pdf("data/Robinson Advisory.docx.pdf")
    text = "\n\n".join(result).replace("\n", " ")
    # chunked_text = chunk_text(text)
    chunked_text = text_splitter.create_documents([text])
    embeded_text = embed_text(chunked_text)

    vectorize(chunked_text, embeded_text, "Robinson")
