import os
import getpass
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing import List, TypedDict
from constant.WebUrls import urls, soup_strainer_classes
from State import State
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import fetch_update_omdb

load_dotenv()

def init_env_vars():
    LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    if not os.environ.get('USER_AGENT'):
        os.environ['USER_AGENT'] = 'FinanceAgent-1.0'
    os.environ['LANGSMITH_TRACING'] = 'true'
    os.environ['LANGSMITH_API_KEY'] = LANGSMITH_API_KEY
    if not os.environ.get('GROQ_API_KEY'):
        os.environ['GROQ_API_KEY'] = GROQ_API_KEY
        
def init_llm():
    try:
        llm = init_chat_model(model='llama3-8b-8192', model_provider='groq')
        print("LLM Initialized")
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None

def init_embeddings_model():
    embeddings_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    return embeddings_model

def init_vector_store():
    embeddings_model = init_embeddings_model()
    persist_directory = './chroma_db'
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    vector_store = Chroma(embedding_function=embeddings_model, persist_directory=persist_directory)
    return vector_store

def load_documents(doc_type: str = 'web'):
    if doc_type == 'web':
        loader = WebBaseLoader(web_paths=(urls), bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=(soup_strainer_classes))))
        docs = loader.load()
        return docs

def split_docs(docs: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def indexing_documents(all_splits: List[Document]):
    vector_store = init_vector_store()
    _ = vector_store.add_documents(documents=all_splits)

def init_prompt():
    prompt = hub.pull('rlm/rag-prompt')
    return prompt

# Defining application steps
def retrieve(state: State) -> dict:
    print(f'Retrieving...........')
    vector_store = init_vector_store()
    print(f'Initialized vector store...')
    retrieved_docs = vector_store.similarity_search(state['question'])
    print(f'Retrieved documents: {retrieved_docs}')
    return {'context': retrieved_docs}

def generate(state: State) -> dict:
    if llm is None:
        return {"answer": "Error initializing LLM"}
    print(f'{llm} is fine..........')
    docs_content = '\n\n'.join([doc.page_content for doc in state['context']])
    prompt = init_prompt()
    messages = prompt.invoke({'question': state['question'], 'context': docs_content})
    response = llm.invoke(messages)
    print(f'Returning generated reponse ...')
    return {'answer': response.content}

init_env_vars()
llm = init_llm()

# Compiling the pipeline
def compile_pipeline():
    print('Creating the state graph')
    graph_builder = StateGraph(State).add_sequence([retrieve, generate]) #pass the llm object.
    print('State graph created...')
    graph_builder.add_edge(START, "retrieve")
    print('Adding an edge for the START...')
    graph = graph_builder.compile()
    print('Returning graph object')
    return graph

