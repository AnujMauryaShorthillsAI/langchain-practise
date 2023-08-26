import os
import warnings
import openai
import langchain
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools.python.tool import PythonREPLTool
from langchain.agents import tool
from datetime import date
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv()) # read local .env file
warnings.filterwarnings("ignore")


class Agents:
    def __init__(self):
        self.configure_api()
    
    def configure_api(self):
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_API_BASE")
    
    # Built In langchain tools
    def use_math_wikipedia_tool(self, question):
        llm = ChatOpenAI(temperature=0, model_kwargs={"engine": "GPT3-5"})

        tools = load_tools(['llm-math', 'wikipedia'], llm=llm)
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=False
        )

        result = agent(question)

        return result
    
    # Using Python Agent
    def use_python_agent(self, question):
        llm = ChatOpenAI(temperature=0, model_kwargs={"engine": "GPT3-5"})

        agent = create_python_agent(
            llm,
            tool=PythonREPLTool(),
            verbose=False
        )

        return agent.run(question)
    
    # Define our own tool
    def own_tool(self, question):
        @tool
        def time(text: str) -> str:
            """Returns todays date, use this for any \
            questions related to knowing todays date. \
            The input should always be an empty string, \
            and this function will always return todays \
            date - any date mathmatics should occur \
            outside this function."""
            return str(date.today())
        
        llm = ChatOpenAI(temperature=0, model_kwargs={"engine": "GPT3-5"})
        agent = initialize_agent(
            [time],
            llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=False
        )

        return agent(question)


if __name__ == "__main__":
    agent = Agents()

    # question = "Tom M. Mitchell is an American computer scientist \
    # and the Founders University Professor at Carnegie Mellon University (CMU)\
    # what book did he write?"
    # res = agent.use_math_wikipedia_tool(question)
    # print(res)


    # customer_list = [["Harrison", "Chase"], 
    #              ["Lang", "Chain"],
    #              ["Dolly", "Too"],
    #              ["Elle", "Elem"], 
    #              ["Geoff","Fusion"], 
    #              ["Trance","Former"],
    #              ["Jen","Ayai"]
    #             ]
    # question = f"""Sort these customers by \
    #     last name and then first name \
    #     and print the output: {customer_list}"""
    
    # # langchain.debug = True
    # res = agent.use_python_agent(question)
    # # langchain.debug = False
    # print(res)


    res = agent.own_tool("what's the date today?")
    print(res)
    