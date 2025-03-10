import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import csv
import shelve
import chromadb
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing import TypedDict, List
from langchain_core.documents import Document
from dotenv import load_dotenv
import fetch_update_omdb
import os
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
import getpass
import os
from langchain.chat_models import init_chat_model

from typing import TypedDict, List
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    llm: ChatGroq  # Add LLM to the state

load_dotenv()

def init_env_vars():
    """Initializes environment variables for LangSmith and Groq API keys."""
    LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not os.environ.get('USER_AGENT'):
        os.environ['USER_AGENT'] = 'FinanceAgent-1.0'
    os.environ['LANGSMITH_TRACING'] = 'true'
    os.environ['LANGSMITH_API_KEY'] = LANGSMITH_API_KEY
    if not os.environ.get('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def init_llm():
    """Initializes the language model (LLM)."""
    try:
        llm = ChatGroq(model_name='llama3-8b-8192', api_key=os.getenv('GROQ_API_KEY'))
        print("LLM Initialized with Groq")
        return llm
    except Exception as e:
        print(f"Error initializing LLM with Groq: {e}")
        return None

def init_embeddings_model():
    """Initializes the embeddings model."""
    embeddings_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    return embeddings_model

def init_vector_store():
    """Initializes the Chroma vector store."""
    embeddings_model = init_embeddings_model()
    persist_directory = './chroma_db'
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    vector_store = Chroma(embedding_function=embeddings_model, persist_directory=persist_directory)
    return vector_store

file_path = "/Users/mohitbhoir/Git/Movie_Recommendation_Chatbot/constant/output_movies.txt"
df = pd.read_csv(file_path, sep="^", quoting=csv.QUOTE_ALL)
df["primaryTitle"] = df["primaryTitle"].fillna("").astype(str)

def prepare_documents_for_splitting(df: pd.DataFrame) -> List[Document]:
    """Prepares a list of Document objects from the DataFrame."""
    docs = []
    for index, row in df.iterrows():
        doc = Document(page_content=row["primaryTitle"])
        docs.append(doc)
    return docs

def split_document_into_chunks(docs: List[Document]) -> List[Document]:
    """Splits documents into smaller chunks for vector storage."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def indexing_documents(all_splits: List[Document], batch_size=5000):
    """Indexes document splits into the vector store in smaller batches."""
    vector_store = init_vector_store()

    # Process documents in chunks
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i:i+batch_size]
        vector_store.add_documents(documents=batch)
        print(f"Inserted batch {i//batch_size + 1}/{(len(all_splits)//batch_size) + 1}")

    print("Indexing completed successfully.")

def init_prompt():
    """Initializes the prompt from Langchain Hub."""
    prompt = hub.pull('rlm/rag-prompt')
    return prompt

def retrieve(state: State) -> dict:
    """Retrieves relevant documents from the vector store based on the question."""
    print(f'Retrieving...........')
    vector_store = init_vector_store()
    print(f'Initialized vector store...')
    retrieved_docs = vector_store.similarity_search(state['question'])
    print(f'Retrieved documents: {retrieved_docs}')
    return {'context': retrieved_docs, 'llm': state['llm']}  # Ensure LLM is passed

def generate(state: State) -> dict:
    """Generates an answer using the LLM and retrieved context."""
    llm = state.get('llm')  # Use .get() to avoid KeyError
    if llm is None:
        return {"answer": "Error: LLM not initialized"}
    print(f'{llm} is fine..........')
    docs_content = '\n\n'.join([doc.page_content for doc in state['context']])
    prompt = init_prompt()
    messages = prompt.invoke({'question': state['question'], 'context': docs_content})
    response = llm.invoke(messages)
    print(f'Returning generated response ...')
    return {'answer': response.content}

init_env_vars()
llm = init_llm() # Initialize llm here

def compile_pipeline(llm): # Pass llm as argument
    """Compiles the Langgraph pipeline."""
    print('Creating the state graph')
    graph_builder = StateGraph(State).add_sequence([retrieve, generate]) # No need to pass llm object here, it will be in state
    print('State graph created...')
    graph_builder.add_edge(START, "retrieve")
    print('Adding an edge for the START...')
    graph = graph_builder.compile()
    print('Returning graph object')
    return graph

if __name__ == "__main__":
    docs = prepare_documents_for_splitting(df)
    all_splits = split_document_into_chunks(docs)
    indexing_documents(all_splits) # Index documents only once

    app = compile_pipeline(llm) # Pass llm to compile_pipeline

    if llm is None:
        print("Error: LLM not initialized. Exiting.")
        exit(1)
        
    inputs = {"question": "Recommend me a brad pitt movie", 'llm': llm} # Pass llm in the input state
    response = app.invoke(inputs)
    print(f"Final Response: {response}")