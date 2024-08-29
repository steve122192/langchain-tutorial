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

#########################
# Simple Agent With Tools
#########################

llm = ChatOpenAI(temperature=0)

#result = llm.invoke([HumanMessage(content="What is the weather in Lansing Michigan today. If you don't know, answer 'I don't know'")])

#print(result.content)




# Add Tool!

search = TavilySearchResults(max_results=2)
#print(search.invoke("what is the weather in Lansing Michigan"))

tools = [search]
model_with_tools = llm.bind_tools(tools)

# response = model_with_tools.invoke([HumanMessage(content="What's the weather in Lansing Michigan?")])

# print(response)

# Get the prompt to use - you can modify this!

prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "What's the weather in Lansing Michigan?"})
#breakpoint()