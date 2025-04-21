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
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY2')  # Update with your Pinecone API key
INDEX_NAME = "movies" # Pinecone index name
CSV_FILE = "constant/merged_tconst.csv"  # Change to your actual file path
PINECONE_ENV = "us-east-1-aws"  # Updated to match your Pinecone environment
BATCH_SIZE = 1000  # Pinecone batch size
EMBEDDING_DIMENSION = 384  # Fixed dimension for MiniLM-L6-v2



# ---- Initialize Pinecone ----
def init_pinecone():
    """Initialize Pinecone and create index if it doesn’t exist."""
    try:
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        return pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"❌ Error initializing Pinecone: {e}")
        traceback.print_exc()
        exit(1)

# ---- Get Existing IDs from Pinecone ----
def get_existing_ids(index):
    """Fetch existing tconst IDs from Pinecone to avoid duplicates."""
    existing_ids = set()
    try:
        # Fetch all IDs in batches
        query_response = index.describe_index_stats()
        total_vectors = query_response['total_vector_count']
        print(f"📌 Found {total_vectors} existing vectors in Pinecone.")

        if total_vectors > 0:
            # Fetch in batches
            limit = 10_000  # Pinecone can handle batch queries up to 10K
            for i in range(0, total_vectors, limit):
                results = index.query(vector=[0.0] * EMBEDDING_DIMENSION, top_k=limit, include_metadata=False)
                existing_ids.update([match['id'] for match in results['matches']])
                
        print(f"✅ Retrieved {len(existing_ids)} existing IDs from Pinecone.")
    except Exception as e:
        print(f"⚠️ Error fetching existing IDs from Pinecone: {e}")
        traceback.print_exc()
    
    return existing_ids

# ---- Preprocess Metadata ----
# ---- Preprocess Metadata (Fixing `NaN` and `float to int` Conversion) ----
def preprocess_metadata(row):
    """Convert metadata fields to appropriate types and handle missing values."""
    try:
        return {
            "tconst": str(row["tconst"]) if pd.notna(row["tconst"]) else "Unknown",
            "primaryTitle": str(row["primaryTitle"]) if pd.notna(row["primaryTitle"]) else "Unknown",
            "originalTitle": str(row["originalTitle"]) if pd.notna(row["originalTitle"]) else "Unknown",
            "isAdult": bool(row["isAdult"]) if pd.notna(row["isAdult"]) else False,

            # ✅ Fix: Convert possible float strings like '2019.0' to integers
            "startYear": int(float(row["startYear"])) if pd.notna(row["startYear"]) else 0,
            "endYear": int(float(row["endYear"])) if pd.notna(row["endYear"]) else 0,
            "runtimeMinutes": int(float(row["runtimeMinutes"])) if pd.notna(row["runtimeMinutes"]) else 0,
            "numVotes": int(float(row["numVotes"])) if pd.notna(row["numVotes"]) else 0,

            "genres": row["genres"].split(",") if pd.notna(row["genres"]) else ["Unknown"],
            "averageRating": float(row["averageRating"]) if pd.notna(row["averageRating"]) else 0.0,
            "actor": row["actor"].split(",") if pd.notna(row["actor"]) else ["Unknown"],
            "actress": row["actress"].split(",") if pd.notna(row["actress"]) else ["Unknown"],
            "director": row["director"].split(",") if pd.notna(row["director"]) else ["Unknown"],
            "producer": row["producer"].split(",") if pd.notna(row["producer"]) else ["Unknown"],
            "writer": row["writer"].split(",") if pd.notna(row["writer"]) else ["Unknown"],
            "description": str(row["description"]) if pd.notna(row["description"]) else "No description available."
        }
    except Exception as e:
        print(f"⚠️ Error processing metadata for row {row.get('tconst', 'UNKNOWN')}: {e}")
        traceback.print_exc()
        return None  # Skip this row if metadata processing fails

# ---- Generate Embeddings ----
def generate_embedding(model, text):
    """Generate a sentence embedding using the specified model."""
    vector = model.encode(text).tolist()
    if len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(f"Embedding dimension mismatch! Expected {EMBEDDING_DIMENSION}, but got {len(vector)}")
    return vector

# ---- Upload Data to Pinecone ----
def upload_to_pinecone(df, index, model, existing_ids):
    """Only upload new data that is not already in Pinecone."""
    vectors_to_upsert = []
    skipped_rows = 0
    new_records = 0

    print("🚀 Processing and uploading new data to Pinecone...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        metadata = preprocess_metadata(row)
        if metadata["tconst"] in existing_ids:
            skipped_rows += 1
            continue  # Skip already uploaded IDs

        # Generate embedding
        text_to_embed = metadata["description"] if metadata["description"] else f"{metadata['primaryTitle']} ({metadata['startYear']})"
        vector = generate_embedding(model, text_to_embed)

        # Prepare for upload
        vectors_to_upsert.append({"id": metadata["tconst"], "values": vector, "metadata": metadata})
        new_records += 1

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

    print(f"✅ Successfully uploaded {new_records} new records to Pinecone!")
    print(f"⚠️ Skipped {skipped_rows} already existing records.")

# ---- Main Execution ----
def main():
    """Main function to run the pipeline."""
    try:
        index = init_pinecone()
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        print(f"📌 Using embedding model with {EMBEDDING_DIMENSION} dimensions.")

        # Load existing IDs from Pinecone
        existing_ids = get_existing_ids(index)

        # Load Data
        df = pd.read_csv(CSV_FILE, na_values=["\\N"], dtype=str, low_memory=False)
        print(f"📂 Loaded {len(df)} movies from {CSV_FILE}.")

        # Process and upload new data only
        upload_to_pinecone(df, index, model, existing_ids)

    except Exception as e:
        print(f"❌ Critical Error: {e}")
        traceback.print_exc()
        exit(1)

def get_vector_store():
    """Returns the Pinecone vector store."""
    logging.info("Initializing Pinecone vector store.")

    # ✅ Use HuggingFaceEmbeddings for LangChain compatibility
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ✅ Ensure Pinecone index is initialized
    pinecone_index = init_pinecone()

    # ✅ Check if the text_key is correct (use 'description' if your Pinecone stores descriptions)
    vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings_model, text_key="description")

    return vector_store

if __name__ == "__main__":
    main()