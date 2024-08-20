import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_dir = './files'

@st.cache_resource
def load_local_files():
    # Loading local files from the files folder
    folder = file_dir
    return [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.pdf')]

@st.cache_resource
def setup_docs():
    files = load_local_files()
    # Load documents
    docs = []
    for file_path in files:
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)
    
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 2, 'fetch_k': 4})
    
    print("Finished loading the documents.")
    return retriever