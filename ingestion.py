import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
load_dotenv()

####################################
# Load Documents
####################################
print('Ingesting...')

pdf_paths = ["C:\\Users\\Steven2R\\af_projects\\langchain-tutorial\\Sample-filled-in-MR.pdf", 
             "C:\\Users\\Steven2R\\af_projects\\langchain-tutorial\\UMNwriteup.pdf",
             "C:\\Users\\Steven2R\\af_projects\\langchain-tutorial\\BCS_Sample_Case_Report.pdf"]

documents = []
for loc in pdf_paths:

    loader = PyPDFLoader(file_path=loc)
    doc = loader.load()
    documents += doc

breakpoint()

####################################
# Split Documents
####################################
print('spliiting')
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=30
)
doc_splits = text_splitter.split_documents(documents)


print(f'created {len(doc_splits)} chunks')

####################################
# Embed & Store
####################################

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
print('storing')
PineconeVectorStore.from_documents(doc_splits, embeddings, index_name=os.environ['INDEX_NAME'])
print('finish')

breakpoint()