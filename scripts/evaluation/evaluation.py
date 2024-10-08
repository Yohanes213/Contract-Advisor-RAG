from scripts.response_generation.response_generation import generate_response
from scripts.retrieval.retriever import retriever
import docx
from langchain_openai import ChatOpenAI
import os
import asyncio
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_similarity,
)
from scripts.evaluation.evaluation_result import visualize_result


load_dotenv()
openai_key = os.getenv("OPENAI_API_KEYS")
pinecone_key = os.getenv("PINECONE_API_KEYS")

os.environ["OPENAI_API_KEY"] = openai_key

retriever = retriever()


def load_evaluation_dataset(docx_path):
    """
    Loads an evaluation dataset from a DOCX file.

    Parameters:
    docx_path (str): Path to the DOCX file containing the questions and answers.

    Returns:
    tuple: Two lists, one with questions and one with corresponding answers.
    """
    doc = docx.Document(docx_path)
    questions = []
    answers = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text.startswith("Q"):
            questions.append(text)
        elif text.startswith("A"):
            answers.append(text)

    return questions, answers


async def runner(questions):
    """
    Generates responses asynchronously for a list of questions.

    Parameters:
    questions (list): A list of questions to generate responses for.

    Returns:
    list: A list of generated responses.
    """
    result = await generate_response(questions)
    return result

questions, ground_truth = load_evaluation_dataset("data/Robinson Q&A.docx")

def evaluate_metrics():
    """
    Evaluates performance metrics for generated responses.

    Returns:
    dict: A dictionary with the evaluation metrics.
    """
    answers = []
    contexts = []
    loop = asyncio.get_event_loop()

    for question in questions:
        result = loop.run_until_complete(runner(question))
        answers.append(result)
        contexts.append(
            [docs.page_content for docs in retriever.get_relevant_documents(question)]
        )

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_similarity,
        ],
    )

    return result


if __name__ == "__main__":
    result = evaluate_metrics()

    print(result)
