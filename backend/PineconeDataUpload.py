import os
from typing import List
from langchain.schema import Document
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import pinecone
import pandas as pd
from dotenv import load_dotenv
import logging
from pinecone import Pinecone

# Ensure sentence-transformers is installed
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.error("Missing 'sentence-transformers' module. Installing it now...")
    os.system("pip install sentence-transformers")
    from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = "movies-actors"
PINECONE_VECTOR_DIMENSION = 384
PINECONE_BATCH_SIZE = 100
USE_HUGGINGFACE_EMBEDDINGS = True  # Change to False to use Pinecone embeddings
INPUT_CSV_FILE_PATH = "/Users/mohitbhoir/Git/Movie_Recommendation_Chatbot/constant/first_5000_movies.csv"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pinecone Initialization ---
pinecone_client = None

def init_pinecone(api_key: str, index_name: str, dimension: int):
    """Initializes Pinecone with proper error handling using the new API."""
    global pinecone_client
    if pinecone_client is None:
        logging.info("Initializing Pinecone client.")
        try:
            pinecone_client = Pinecone(api_key=api_key)
        except Exception as e:
            logging.error(f"Failed to initialize Pinecone: {e}")
            raise

    index_list = pinecone_client.list_indexes().names()
    
    if index_name in index_list:
        logging.info(f"Initializing existing Pinecone index: {index_name}")
        return pinecone_client.Index(index_name)
    else:
        logging.error(f"Index {index_name} does not exist. Please create it manually before running this script.")
        raise ValueError(f"Index {index_name} does not exist.")

def init_embeddings_model():
    """Initializes either HuggingFace or Pinecone embeddings model."""
    if USE_HUGGINGFACE_EMBEDDINGS:
        logging.info("Using HuggingFace embeddings model.")
        return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    else:
        logging.info("Using Pinecone built-in embeddings model.")
        return None  # Pinecone supports upserting raw text, embedding on their servers

def init_pinecone_vector_store(api_key: str, index_name: str, dimension: int):
    """Initializes the Pinecone vector store."""
    logging.info("Initializing Pinecone vector store.")
    embeddings_model = init_embeddings_model()
    pinecone_index = init_pinecone(api_key, index_name, dimension)
    vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings_model, text_key="page_content")
    return vector_store

def indexing_documents_pinecone(all_splits: List[Document], api_key: str, index_name: str, dimension: int, batch_size: int = 500):
    """Indexes document splits into the Pinecone vector store in smaller batches, respecting free-tier limits."""
    logging.info("Starting Pinecone indexing.")
    pinecone_index = init_pinecone(api_key, index_name, dimension)  # Pass dimension
    embeddings_model = init_embeddings_model()

    total_batches = (len(all_splits) // batch_size) + (1 if len(all_splits) % batch_size else 0)
    
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i:i + batch_size]
        if not batch:
            logging.warning(f"Skipping empty batch {i // batch_size + 1}/{total_batches}")
            continue

        try:
            contents = [doc.page_content for doc in batch]
            embeddings = embeddings_model.embed_documents(contents)

            vectors = [
                (str(doc.metadata["tconst"]), embedding, doc.metadata)
                for doc, embedding in zip(batch, embeddings)
            ]
            if vectors:
                pinecone_index.upsert(vectors=vectors)
                logging.info(f"Inserted batch {i // batch_size + 1}/{total_batches} ({len(batch)} documents)")
            else:
                logging.warning(f"Skipping batch {i // batch_size + 1} due to missing vectors.")
        except Exception as e:
            logging.error(f"Error inserting batch {i // batch_size + 1}: {e}")
            logging.exception(e)

    logging.info("Indexing completed successfully.")

def load_documents_from_csv(file_path: str) -> List[Document]:
    """Loads documents from the processed CSV file with NaN handling."""
    logging.info(f"Loading documents from {file_path}.")
    df = pd.read_csv(file_path)

    # Handle missing columns gracefully
    required_columns = {"primaryTitle", "startYear", "genres", "tconst"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    # Replace NaN values with appropriate defaults
    df.fillna({"primaryTitle": "Unknown Title", "startYear": "Unknown Year", "genres": "Unknown Genre"}, inplace=True)
    df.fillna("", inplace=True)  # Replace remaining NaNs with empty string

    documents = []
    for _, row in df.iterrows():
        metadata = {col: str(row[col]) if pd.notna(row[col]) else "N/A" for col in df.columns if col != "page_content"}
        metadata["tconst"] = str(row["tconst"])  # Ensure it's a string
        page_content = f"{metadata['primaryTitle']} ({metadata['startYear']}) - {metadata['genres']}"

        documents.append(Document(page_content=page_content, metadata=metadata))

    logging.info(f"Loaded {len(documents)} documents from CSV with NaN handling.")
    return documents

def upload_data_to_pinecone():
    """Loads processed data and uploads to Pinecone in batches."""
    if not PINECONE_API_KEY:
        logging.error("Pinecone API key not set.")
        return

    documents = load_documents_from_csv(INPUT_CSV_FILE_PATH)
    logging.info(f"Total documents loaded: {len(documents)}")
    
    if not documents:
        logging.error("No valid documents found. Exiting upload process.")
        return

    try:
        indexing_documents_pinecone(documents, PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_BATCH_SIZE)
    except Exception as e:
        logging.error(f"Error during indexing: {e}", exc_info=True)

def get_vector_store():
    """Returns the Pinecone vector store."""
    logging.info("Initializing Pinecone vector store.")
    embeddings_model = init_embeddings_model()
    pinecone_index = init_pinecone(PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_VECTOR_DIMENSION)  # Pass dimension
    vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings_model, text_key="page_content")
    return vector_store

if __name__ == "__main__":
    upload_data_to_pinecone()