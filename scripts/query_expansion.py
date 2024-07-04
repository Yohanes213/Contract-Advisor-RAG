from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEYS")

client = OpenAI(
  api_key=openai_key
)

def expand_query_hypothetical(query, model="gpt-3.5-turbo"):

    messages = [
        {
            "role": "system",
            "content":(
                "You are a knowledgeable contract assistant. For the given question, provide a hypothetical example answer "
                "that one might find in an Advisory Services Agreement document. Ensure the example is clear, concise, and relevant to the query."
                "Answer in one sentence and make it like you have an information."
            )
        
        },
        {"role": "user", "content": query}
    ] 

    response = client.chat.completions.create(
        model = model,
        messages = messages,
        temperature = 0
    )

    return response.choices[0].message.content


def expand_query_multiple(query, model="gpt-3.5-turbo"):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful expert in contract assistance. Users are asking questions about a contract document. "
                "For the provided question, suggest up to five additional related questions to help them find the information they need. "
                "Ensure the questions are short, cover different aspects of the topic, and are directly related to the original question. They should be related to the original query."
                "Output each question on a new line without numbering them."
            )
        },
        {"role": "user", "content": query}
    ] 

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    query = "Who are the parties to the Agreement and what are their defined names?"
    query1 = "What is the termination notice?"

    augmented_queries = expand_query_hypothetical(query)

    print(augmented_queries)
