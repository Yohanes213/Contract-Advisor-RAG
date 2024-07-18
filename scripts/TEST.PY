__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join('.', 'db.sqlite3'),
    }
}
import json


import chromadb

import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

# Accepted file formats for that can be stored in
# a vector database instance
from autogen.retrieve_utils import TEXT_FORMATS
import gradio as gr
import multiprocessing as mp




config_list = [
    {"model": "gpt-3.5-turbo-0125", "api_key": os.getenv('OPENAI_API_KEYS'), "api_type": "openai"},
]

assert len(config_list) > 0
print("models to use: ", [config_list[i]["model"] for i in range(len(config_list))])

# Initialize Agents
def initialize_agents(config_list, docs_path=None):
    # 1. create an RetrieveAssistantAgent instance named "assistant"
    assistant = RetrieveAssistantAgent(
        name="assistant",
        system_message="""
        You are an AI expert specialized in contract analysis and advisory services.
        Your goal is to assist users by providing precise and helpful answers based on a contract document.
        The document in question is an Advisory Services Agreement.
        Given the user query, generate a well-structured and informative response by extracting relevant information from the document.
        Ensure the response is concise, accurate, and directly addresses the query.
        """,
        llm_config={
            "timeout": 600,
            "cache_seed": 42,
            "config_list": config_list,
        },
    )

    # 2. create the RetrieveUserProxyAgent instance named "ragproxyagent"
    # By default, the human_input_mode is "ALWAYS", which means the agent will ask for human input at every step. We set it to "NEVER" here.
    # `docs_path` is the path to the docs directory. It can also be the path to a single file, or the url to a single file. By default,
    # it is set to None, which works only if the collection is already created.
    # `task` indicates the kind of task we're working on. In this example, it's a `code` task.
    # `chunk_token_size` is the chunk token size for the retrieve chat. By default, it is set to `max_tokens * 0.6`, here we set it to 2000.
    # `custom_text_types` is a list of file types to be processed. Default is `autogen.retrieve_utils.TEXT_FORMATS`.
    # This only applies to files under the directories in `docs_path`. Explicitly included files and urls will be chunked regardless of their types.
    # In this example, we set it to ["non-existent-type"] to only process markdown files. Since no "non-existent-type" files are included in the `websit/docs`,
    # no files there will be processed. However, the explicitly included urls will still be processed.
    ragproxyagent = RetrieveUserProxyAgent(
        name="ragproxyagent",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        retrieve_config={
            "task": "qa",
            "docs_path": [
                # "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
                # "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md",
                # os.path.join(os.path.abspath(""), "..", "website", "docs"),
                '../data/Robinson Advisory.docx.pdf'
            ],
            "custom_text_types": ["non-existent-type"],
            "chunk_token_size": 1000,
            "model": config_list[0]["model"],
            #"client": chromadb.PersistentClient(path="/tmp/chromadb"),  # deprecated, use "vector_db" instead
            "vector_db": "chroma",  # to use the deprecated `client` parameter, set to None and uncomment the line above
            "overwrite": True,  # set to True if you want to overwrite an existing collection
            #"chunk_mode": "one_line",
            #"must_break_at_empty_line": False,
            #"embedding_model": embed_model,
            #"update_context": True,
            #"custom_text_split_function": recur_spliter.split_text,
        },
        code_execution_config=False,  # set to False if you don't want to execute the code
    )
    return assistant, ragproxyagent

# Initialize Chat
def initiate_chat(config_list, problem, queue, n_results=3):
    ...
    assistant.reset()
    try:
        ragproxyagent.a_initiate_chat(
            assistant, problem=problem, silent=False, n_results=n_results
        )
        messages = ragproxyagent.chat_messages
        messages = [messages[k] for k in messages.keys()][0]
        messages = [m["content"] for m in messages if m["role"] == "user"]
        print("messages: ", messages)
    except Exception as e:
        messages = [str(e)]
    queue.put(messages)

# Wrap AutoGen part into a function
def chatbot_reply(input_text):
    """Chat with the agent through terminal."""
    queue = mp.Queue()
    process = mp.Process(
        target=initiate_chat,
        args=(config_list, input_text, queue),
    )
    process.start()
    try:
        messages = queue.get(timeout=20)
    except Exception as e:
        messages = [str(e) if len(str(e)) > 0 else "Invalid Request to OpenAI, please check your API keys."]
    finally:
        try:
            process.terminate()
        except:
            pass
    return messages

...

# Set up UI with Gradio
with gr.Blocks() as demo:
    ...
    assistant, ragproxyagent = initialize_agents(config_list)

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "autogen.png"))),
        # height=600,
    )

    txt_input = gr.Textbox(
        scale=4,
        show_label=False,
        placeholder="Enter text and press enter",
        container=False,
    )

    with gr.Row():
        txt_model = gr.Dropdown(
            label="Model",
            choices=[
                #"gpt-4",
                #"gpt-35-turbo",
                "gpt-3.5-turbo",
            ],
            allow_custom_value=True,
            value="gpt-3.5-turbo",
            container=True,
        )
        txt_oai_key = gr.Textbox(
            label="OpenAI API Key",
            placeholder="Enter key and press enter",
            max_lines=1,
            show_label=True,
            value=os.environ.get("OPENAI_API_KEY", ""),
            container=True,
            type="password",
        )
        ...

    clear = gr.ClearButton([txt_input, chatbot])

...

if __name__ == "__main__":
    demo.launch(share=True)