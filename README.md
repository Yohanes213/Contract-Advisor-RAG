# Contract-Advisor-RAG: Building a High-Precision Legal Contract LLM App

## Overview

This project aims to develop a Retrieval Augmented Generation (RAG) system for Contract Q&A, forming the basis for Lizzy AI's fully autonomous contract lawyer. The system combines the power of language models with external data retrieval to answer questions about contracts with high precision.


## Installation 
1. Clone the repo

```bash
git clone github.com/Yohanes213/Contract-Advisor-RAG
cd Contract-Advisor-RAG
```
2. Install the dependencies
   
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEYS=<your_openai_api_key>
export PINECONE_API_KEY=<your_pinecone_api_key>
export LANGCHAIN_API_KEY=<your_langchain_api_key>
export COHERE_API_KEYS=<your_cohere_api_key>
```

## Usage

Run the Streamlit App

``` bash
streamlit run app.py
```

## Contributing

Contributions are welcome! Please create an issue or pull request for any feature requests or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

