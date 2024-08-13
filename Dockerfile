# Use an official Python image as a base
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file
COPY requirements.txt.

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the application code
COPY..

# Set the environment variables
ENV OPENAI_API_KEYS=${OPENAI_API_KEYS}
ENV PINECONE_API_KEY=${PINECONE_API_KEY}
ENV LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
ENV COHERE_API_KEYS=${COHERE_API_KEYS}

# Expose the port for the Streamlit app
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]