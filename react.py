import os
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain.agents import create_tool_calling_agent

load_dotenv()

#######################
# ReAct Agent
######################

llm = ChatOpenAI(temperature=0)
search = TavilySearchResults(max_results=2)

def monthly_wage(salary):
    """
    Takes salary input and divides by 12 to return monthly wage
    """
    return float(salary)/12

wage =  Tool(
            name="Get monthly wage based on salary",
            func=monthly_wage,
            description="useful for when you need to determine monthly wage based on annual salary. Takes integer as input with no symbols or commas",
        )

tools = [search, wage]

template = """
        given the Job title  about a person, I want you to create:
        1. a description of what their day at work might look like
        2. How much money they might make per year
        3. How much money they might make per month
        Job Title: {title}
        Use tools to help get an answer
    """
prompt_template = PromptTemplate(
    template=template, input_variables=["title"]
)
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

result = agent_executor.invoke(
    input={"input": prompt_template.format_prompt(title="Zookeeper")}
)
