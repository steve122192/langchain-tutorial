import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


####################################
# Without RAG
####################################
llm = ChatOpenAI()

query = "What is the patients name?"

chain = PromptTemplate(template=query) | llm
result = chain.invoke(input={})
print('----------------------------------------------------')
print(result.content)
print('----------------------------------------------------')

####################################
# Setup Retreiever
####################################

embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'], embedding=embeddings
)
retriever = vectorstore.as_retriever()

####################################
# Pre-Built RAG Chain
####################################
# pre-written prompt
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
# combine retrieved docs
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
# setup retrieval chain
retrieval_chain = create_retrieval_chain(
    retriever=retriever, combine_docs_chain=combine_docs_chain
)

result = retrieval_chain.invoke(input={"input": query})
print('----------------------------------------------------')
print(result['answer'])
print('----------------------------------------------------')


####################################
# Custom Rag Chain
####################################

template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maxium and keep the answer as concise a possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

custom_rag_prompt =  PromptTemplate.from_template(template)

rag_chain = (
    {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)
res = rag_chain.invoke(query)

print('----------------------------------------------------')
print(res)
print('----------------------------------------------------')
