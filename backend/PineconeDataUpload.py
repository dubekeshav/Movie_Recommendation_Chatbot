import os
from typing import List
from langchain.schema import Document
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import pinecone
import pandas as pd
from dotenv import load_dotenv
import logging

load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = "movies-actors"
PINECONE_VECTOR_DIMENSION = 384
PINECONE_BATCH_SIZE = 100
INPUT_CSV_FILE_PATH = "formatted_movies_per_movie.csv"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pinecone Initialization ---
def init_pinecone(api_key: str, index_name: str, dimension: int):
    """Initializes Pinecone."""
    logging.info("Initializing Pinecone client.")
    pc = pinecone.Pinecone(api_key=api_key)
    if index_name not in [i.name for i in pc.list_indexes()]:
        logging.info(f"Creating Pinecone index: {index_name}")
        pc.create_index(index_name, dimension=dimension, metric="cosine")
    else:
        logging.info(f"Using existing Pinecone index: {index_name}")
    return pc.Index(index_name)

def init_embeddings_model():
    """Initializes the embeddings model."""
    logging.info("Initializing HuggingFace embeddings model.")
    embeddings_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    return embeddings_model

def init_pinecone_vector_store(api_key: str, index_name: str, dimension: int):
    """Initializes the Pinecone vector store."""
    logging.info("Initializing Pinecone vector store.")
    embeddings_model = init_embeddings_model()
    pinecone_index = init_pinecone(api_key, index_name, dimension)
    vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings_model, text_key="page_content") #Change Here.
    return vector_store

def indexing_documents_pinecone(all_splits: List[Document], api_key: str, index_name: str, dimension: int, batch_size: int):
    """Indexes document splits into the Pinecone vector store in smaller batches."""
    logging.info("Starting Pinecone indexing.")
    vector_store = init_pinecone_vector_store(api_key, index_name, dimension)

    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i:i + batch_size]
        try:
            for doc in batch:
                if doc.metadata:
                    for key, value in doc.metadata.items():
                        doc.metadata[key] = str(value) #Explicit string conversion.
            vector_store.add_documents(documents=batch)
            logging.info(f"Inserted batch {i // batch_size + 1}/{(len(all_splits) // batch_size) + 1}")
        except Exception as e:
            logging.error(f"Error inserting batch {i // batch_size + 1}: {e}")
            logging.exception(e)

    logging.info("Indexing completed successfully.")

def load_documents_from_csv(file_path: str) -> List[Document]:
    """Loads documents from the processed CSV file."""
    logging.info(f"Loading documents from {file_path}.")
    df = pd.read_csv(file_path)
    documents = []
    for _, row in df.iterrows():
        metadata = {col: row[col] for col in df.columns if col != "page_content"}
        documents.append(Document(page_content=row["page_content"], metadata=metadata))
    logging.info(f"Loaded {len(documents)} documents from CSV.")
    return documents

def upload_data_to_pinecone():
    """Loads processed data and uploads to Pinecone."""
    if not PINECONE_API_KEY:
        logging.error("Pinecone API key not set.")
        return

    documents = load_documents_from_csv(INPUT_CSV_FILE_PATH)
    
    # print(documents.isna().sum())
    indexing_documents_pinecone(documents, PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_VECTOR_DIMENSION, PINECONE_BATCH_SIZE)
    
def get_vector_store():
    """Returns the Pinecone vector store."""
    logging.info("Initializing Pinecone vector store.")
    embeddings_model = init_embeddings_model()
    pinecone_index = init_pinecone(PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_VECTOR_DIMENSION)
    vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings_model, text_key="page_content")
    return vector_store

if __name__ == "__main__":
    upload_data_to_pinecone()