from chains_graph import retrieval_chain, agent_executor, question_router
from summary_chain import summary_chain
import operator
from typing import Annotated, List, Literal, TypedDict, Tuple

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langchain_community.document_loaders import PyPDFLoader

####################################
# Define State
####################################
class MyState(TypedDict):
    input: str
    router_response: str
    answer: str
    documents: List[Document]


####################################
# Define Nodes
####################################
def route(state: MyState):
    response = question_router.invoke(input={"question": state["input"]})
    return {"router_response": response.datasource}

def retrieve(state: MyState):
    response = retrieval_chain.invoke(input={"input": state["input"]})
    return {"answer": response['answer'], "documents": [(doc.metadata['source'], doc.metadata['page']) for doc in response['context']]}

def search(state: MyState):
    response = agent_executor.invoke(input={"input": state["input"]})
    return {"answer": response['output']}
    
def summarize(state: MyState):
    pdf_path = "C:\\Users\\Steven2R\\af_projects\\langchain-tutorial\\Sample-filled-in-MR.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()
    response = summary_chain.invoke({"context": docs})
    return {"answer": response}

####################################
# Define Conditional Edge
####################################  
def router(state: MyState) -> Literal["vectorstore", "websearch"]:
    if state["router_response"] == 'vectorstore':
        return "retrieve"
    if state["router_response"] == 'websearch':
        return "search"  
    if state["router_response"] == 'summary':
        return 'summarize'

####################################
# Build Graph
####################################  
graph = StateGraph(MyState)
# nodes
graph.add_node("route", route)  
graph.add_node("retrieve", retrieve) 
graph.add_node("search", search)
graph.add_node("summarize", summarize)

# edges
graph.set_entry_point("route")
graph.add_conditional_edges("route",
                            router,
    {
        "retrieve": "retrieve",
        "search": "search",
        "summarize": "summarize"
    }
    )
graph.add_edge("retrieve", END)
graph.add_edge("search", END)
graph.add_edge("summarize", END)

# compile
app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="tutorial_graph.png")



response = app.invoke(input={"input": "Summarize the document"})
print(response['answer'])
if response['router_response'] == 'vectorstore':
    print(response['documents'])