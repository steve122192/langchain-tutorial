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
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import Literal
from langchain_community.document_loaders import PyPDFLoader

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
load_dotenv()

llm = ChatOpenAI(temperature=0)

# Define prompt
prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following:\\n\\n{context}")]
)

# Instantiate chain
summary_chain = create_stuff_documents_chain(llm, prompt)



pdf_path = "C:\\Users\\Steven2R\\af_projects\\langchain-tutorial\\Sample-filled-in-MR.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()
response = summary_chain.invoke({"context": docs})
#print(response)