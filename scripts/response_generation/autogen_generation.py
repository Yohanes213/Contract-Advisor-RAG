__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import json

import chromadb

import autogen
from scripts.retrieval.retriever import retriever


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join('.', 'db.sqlite3'),
    }
}

config_list = [
    {"model": "gpt-3.5-turbo-0125", "api_key": os.getenv('OPENAI_API_KEYS'), "api_type": "openai"},
]

assert len(config_list) > 0
print("models to use: ", [config_list[i]["model"] for i in range(len(config_list))])

worker = autogen.AssistantAgent(
name="worker",
llm_config={"config_list": config_list, "cache_seed":42},
code_execution_config=False,

# the default system message of the AssistantAgent is overwritten here
system_message="You are a helpful AI worker. You are a worker agent who does exactly what you are told to do.",
)


retriever_obj = retriever()

def ask_worker(message):
     # context = ""
      #if retrival == "TRUE":
       # try:
      print("retrival started")
      relevant_docs = retriever_obj.invoke(message)
      context = "\n " + "Answer the question(message) from this important information to achieve the tasks: " + vars(relevant_docs[0])['page_content']# + " " + vars(relevant_docs[1])['page_content']
    #  context = "\n " + "Answer the question(message) from this important information to achieve the tasks: " + str(retriever_obj.invoke(message))
      print("retrival ended")

      worker.llm_config = {"config_list": config_list, "response_format":{ "type": "json_object" }}
      
      prepared_message = [{'content':  message + context ,  'role': 'assistant'}]

      print(" ************** Fully Prepared Worker Context *************************")
      print(prepared_message[0]["content"])
      print(" ************** Fully Prepared Worker Context *************************")

      worker.llm_config = {"config_list": config_list, "response_format":{ "type": "json_object" }}
      result = worker.generate_reply(messages=prepared_message)

      return result
    

# questions, ground_truth = load_evaluation_dataset("data/Robinson Q&A.docx")

assistant = autogen.AssistantAgent(
    name="assistant",
    # system_message="""
    
    # Your goal is to assist users by providing precise and helpful answers based on a contract document.
    # The document in question is an Advisory Services Agreement.
    # Given the user query, generate a well-structured and informative response by extracting relevant information from the document.
    # Ensure the response is concise, accurate, and directly addresses the query.
    # """,

    system_message = """
    You are an AI expert specialized in contract analysis and advisory services. Your primary objective is to plan steps to appropriately solve users'\
     requests or tasks using worker function calling.

    The main problems you will be solving include:
    - Answering contract questions

    Your tasks:
    1. Use the `ask_worker` function to delegate each step, ensuring it returns the results upon task completion.
    2. VERY IMPORTANT: Before you return any result ensure that you do not need worker skills to perform a task.

    Key guidelines:
    - ALWAYS use the ask_worker function for the task and send it only the most important instructions and commands.
    - FORCE the ask_worker function to only give you the answer for your request without adding explanation or guidance what it achieved.

    Communication:
    - Your answer should be constructed in the given language the user used to ask.
    - Before responding to the user make sure you used all the skills the worker function provides.
    - When communicating with the Worker make sure to send The most Important context to it.

    """,

    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
        "functions": [
             {
                        "name": "ask_worker",
                        "description": "ask worker to:  1. generate questions for requirement gathering, 2. retrieve relevant contract based on the topic sent as message, 3. Return the answer based on the message and the retrived relevant document",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "question to ask worker. Make sure the question include enough context, such as the required task. The worker does not know the conversation between you and the user, unless you share the conversation with the worker. Ensure the original question of the user is also passed as context such that the worker is generating intermediate steps necessary to address it. It is VERY IMPORTANT that you provide a clear and autoritative feedback to the worker to stay focused on answering the original question."
                                    },
                                # "message_type": {
                                #     "type": "string",
                                #     "description": "Type of work you want it to achieve. Options are: 'REQUIREMENT GATHERING','SIMPLE QUESTION ANSWERING'"
                                # },
                                # "retrival":{
                                #     "type": "string",
                                #     "description": "Becomes 'TRUE' or 'FALSE' which would indicate to perform retrieval of relevant articles from the law to achieve the task"
                                # },
                            },
                                
                        }
                        }
             
        ],
    },
)


user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    #is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    is_termination_msg=lambda x: "content" in x and x["content"] is not None and x["content"].rstrip().endswith("TERMINATE"),
    # code_execution_config={"work_dir": "planning"},
    function_map={"ask_worker": ask_worker },
)

def chat_with_agent(message):
  # combined_message, lang = translate_to_dest(message,"en")
  user_proxy.initiate_chat(
      assistant,
      #clear_history=False,
     # silent=True,
      message=message
  )
