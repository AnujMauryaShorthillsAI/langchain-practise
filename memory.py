import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

"""
 Can Use Following Memory As Well
    1. ConversationBufferMemory
    2. ConversationBufferWindowMemory
"""

_ = load_dotenv(find_dotenv()) # read local .env file

class ConversationChainWithMemory:

    def __init__(self) -> None:
        self.initialize_api()
        self.initialize_components()
    
    def initialize_api(self):
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_API_BASE")
    
    def initialize_components(self):
        llm = ChatOpenAI(temperature=0, model_kwargs = {"engine" : "GPT3-5"})
        self.memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
        self.conversation = ConversationChain(llm=llm, memory=self.memory, verbose=False)
    
    def get_completion(self, schedule):
        # Explicitly Saving Context
        self.memory.save_context({"input": "Hello"}, {"output": "What's up"})
        self.memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
        self.memory.save_context({"input": "What is on the schedule today?"}, {"output": f"{schedule}"})

        result = self.conversation.predict(input="What would be a good demo to show?")
        return result


# Conversation Summary Memory
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."


chat = ConversationChainWithMemory()
res = chat.get_completion(schedule)
print(res)
