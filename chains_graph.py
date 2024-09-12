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

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
load_dotenv()

llm = ChatOpenAI(temperature=0)

####################################
# Define Chains
####################################

# React Chain
search = TavilySearchResults(max_results=2)

tools = [search]

# template = """
#         Use provided tools to search for information about a users question
#         User Question: {question}
#     """
# prompt_template = PromptTemplate(
#     template=template, input_variables=["question"]
# )
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Rag Chain
# Setup retriever
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'], embedding=embeddings
)
retriever = vectorstore.as_retriever()
# pre-written prompt
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
# combine retrieved docs
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
# setup retrieval chain
retrieval_chain = create_retrieval_chain(
    retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
)

# Router Chain

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch", "summary"] = Field(
        description="Given a user question choose to route it to web search, vectorstore or summarizer.",
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore, web search, or doc summarizer.
The vectorstore contains medical documents related to a specific patient or event.
Use the vectorstore for questions on these topics. For any other questions, use web-search. """
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: {question}"),
    ]
)

question_router = route_prompt | structured_llm_router

