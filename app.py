import streamlit as st
import asyncio

from scripts.response_generation.response_generation import generate_response
from scripts.main import run
from scripts.response_generation.autogen_generation import ask_worker

# from scripts.query_utils import querying, embed_text


async def main():
    st.title("QA Contract")

    # Initialize the session state to store chat history if not already done
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        user_query = st.text_input("Enter your question:")

    if st.sidebar.button("See Result") and user_query:
        # embedded_text = await embed_text(user_query)
        # result = await querying(embedded_text)

        # retrieved_docs = [match['metadata']['text'] for match in result['matches']]

        # prompt = f"User question: {user_query}\n\n"
        # for i, doc in enumerate(retrieved_docs, 1):
        #     prompt += f"Document {i}: {doc}\n\n"

        # prompt += "Based on the above documents, answer the user's question."
        # print(prompt)

        response = ask_worker(user_query)

        #response = await generate_response(user_query)

        # Add the new question and response to the chat history
        st.session_state.chat_history.append({"user": user_query, "ai": response})

    # Display the chat history
    for chat in st.session_state.chat_history:
        st.write(f"**User:** {chat['user']}")
        st.write(f"**AI:** {chat['ai']}")


if __name__ == "__main__":
    asyncio.run(main())
