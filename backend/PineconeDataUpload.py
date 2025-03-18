import os
from typing import List
from langchain.schema import Document
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import pinecone
import pandas as pd
from dotenv import load_dotenv
import logging
import pinecone
import pandas as pd
import time
import traceback
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Ensure sentence-transformers is installed
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.error("Missing 'sentence-transformers' module. Installing it now...")
    os.system("pip install sentence-transformers")
    from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY2')
PINECONE_API_KEY = "pcsk_2mh5kr_AdiVPKgc8DrhdkL7gaGzJbFdGqKPkRuvkgzXP3xkBdYaX5DCsgPYJriSfyriJwB"
INDEX_NAME = "movies"
CSV_FILE = "constant/merged_tconst.csv"  # Change to your actual file path
PINECONE_ENV = "us-east-1-aws"  # Updated to match your Pinecone environment
BATCH_SIZE = 500  # Pinecone batch size
EMBEDDING_DIMENSION = 384  # Fixed dimension for MiniLM-L6-v2



# ---- Initialize Pinecone (New API) ----
def init_pinecone():
    """Initialize Pinecone and create index if it doesn’t exist."""
    try:
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

        # Check if index exists
        existing_indexes = pc.list_indexes().names()
        if INDEX_NAME not in existing_indexes:
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust to your AWS region
            )
            print(f"Created Pinecone index: {INDEX_NAME} with {EMBEDDING_DIMENSION} dimensions")

        return pc.Index(INDEX_NAME)

    except Exception as e:
        print(f"❌ Error initializing Pinecone: {e}")
        traceback.print_exc()
        exit(1)  # Stop script if Pinecone fails to initialize

# ---- Load Embedding Model ----
def load_embedding_model():
    """Load sentence-transformers model with fixed embedding dimension, with error handling."""
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        if model.get_sentence_embedding_dimension() != EMBEDDING_DIMENSION:
            raise ValueError(f"Model dimension mismatch! Expected {EMBEDDING_DIMENSION}, but got {model.get_sentence_embedding_dimension()}")
        return model
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        traceback.print_exc()
        exit(1)  # Stop script if embedding model fails

# ---- Preprocess Metadata (Fixing `None` Values) ----
def preprocess_metadata(row):
    """Convert metadata fields to appropriate types and handle missing values."""
    try:
        return {
            "tconst": row["tconst"],
            "primaryTitle": row["primaryTitle"],
            "originalTitle": row["originalTitle"],
            "isAdult": bool(row["isAdult"]),
            "startYear": int(row["startYear"]) if pd.notna(row["startYear"]) else 0,  # Default to 0
            "endYear": int(row["endYear"]) if pd.notna(row["endYear"]) else 0,
            "runtimeMinutes": int(row["runtimeMinutes"]) if pd.notna(row["runtimeMinutes"]) else 0,
            "genres": row["genres"].split(",") if pd.notna(row["genres"]) else ["Unknown"],
            "averageRating": float(row["averageRating"]) if pd.notna(row["averageRating"]) else 0.0,  # Default to 0.0
            "numVotes": int(row["numVotes"]) if pd.notna(row["numVotes"]) else 0,  # Default to 0
            "actor": row["actor"].split(",") if pd.notna(row["actor"]) else ["Unknown"],
            "actress": row["actress"].split(",") if pd.notna(row["actress"]) else ["Unknown"],
            "director": row["director"].split(",") if pd.notna(row["director"]) else ["Unknown"],
            "producer": row["producer"].split(",") if pd.notna(row["producer"]) else ["Unknown"],
            "writer": row["writer"].split(",") if pd.notna(row["writer"]) else ["Unknown"],
            "description": row["description"] if pd.notna(row["description"]) else "No description available."
        }
    except Exception as e:
        print(f"⚠️ Error processing metadata for row {row.get('tconst', 'UNKNOWN')}: {e}")
        traceback.print_exc()
        return None  # Skip this row if metadata processing fails

# ---- Generate Embeddings ----
def generate_embedding(model, text):
    """Generate a sentence embedding using the specified model, with error handling."""
    try:
        vector = model.encode(text).tolist()
        
        # Ensure embedding is always 384 dimensions
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(f"Embedding dimension mismatch! Expected {EMBEDDING_DIMENSION}, but got {len(vector)}")
        
        return vector
    except Exception as e:
        print(f"⚠️ Error generating embedding for text: '{text[:50]}...' - {e}")
        traceback.print_exc()
        return None  # Skip this embedding

# ---- Upload Data to Pinecone ----
def upload_to_pinecone(df, index, model):
    """Process and upload data to Pinecone in batches, with error handling."""
    vectors_to_upsert = []
    skipped_rows = 0

    print("Processing and uploading data to Pinecone...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        metadata = preprocess_metadata(row)
        if metadata is None:
            skipped_rows += 1
            continue  # Skip if metadata processing failed

        # Use description for embedding, fallback to title + year if empty
        text_to_embed = metadata["description"] if metadata["description"] else f"{metadata['primaryTitle']} ({metadata['startYear']})"
        vector = generate_embedding(model, text_to_embed)
        if vector is None:
            skipped_rows += 1
            continue  # Skip if embedding generation failed

        # Add to batch
        vectors_to_upsert.append({"id": metadata["tconst"], "values": vector, "metadata": metadata})

        # Upload in batches
        if len(vectors_to_upsert) >= BATCH_SIZE:
            try:
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []  # Clear batch
                time.sleep(1)  # Prevent rate limits
            except Exception as e:
                print(f"❌ Error uploading batch to Pinecone: {e}")
                traceback.print_exc()

    # Upload remaining records
    if vectors_to_upsert:
        try:
            index.upsert(vectors=vectors_to_upsert)
        except Exception as e:
            print(f"❌ Error uploading final batch to Pinecone: {e}")
            traceback.print_exc()

    print(f"✅ Data successfully uploaded to Pinecone with {EMBEDDING_DIMENSION} dimensions!")
    print(f"⚠️ Skipped {skipped_rows} rows due to errors.")

# ---- Main Execution ----
def main():
    """Main function to run the pipeline."""
    try:
        index = init_pinecone()
        model = load_embedding_model()

        print(f"Using embedding model with {EMBEDDING_DIMENSION} dimensions.")

        # Load Data
        df = pd.read_csv(CSV_FILE, na_values=["\\N"], low_memory=False)  # Fix DtypeWarning
        print(f"Loaded {len(df)} movies from {CSV_FILE}.")

        # Process and upload to Pinecone
        upload_to_pinecone(df, index, model)

    except Exception as e:
        print(f"❌ Critical Error: {e}")
        traceback.print_exc()
        exit(1)  # Stop script if an unexpected error occurs

if __name__ == "__main__":
    main()