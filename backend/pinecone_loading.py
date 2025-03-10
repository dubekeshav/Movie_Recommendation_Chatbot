import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import ServerlessSpec, Pinecone
import time
from dotenv import load_dotenv
import json
from io import StringIO

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create an index
index_name = "movies-actors"
if index_name not in pc.list_indexes().names():  # Check if index exists
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled",
        tags={
            "environment": "development"
        }
    )
    while index_name not in pc.list_indexes().names():  # Wait for index creation
        time.sleep(1)

# Connect to the index
index = pc.Index(index_name)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load movie dataset in chunks
output_directory = "constant"
absolute_path = os.path.abspath(output_directory)
movie_file = os.path.join(absolute_path, 'movies_transformed.csv')

vectorized_data = []# Initialize an empty list to store vectorized data
total_size = 0
VECTOR_SIZE = 384 * 4
exceptions = []

def process_row(row_str, line_num):
    """Processes a row using pandas logic first, then delimiter counting."""
    try:
        # Try pandas logic with quoting
        row_df = pd.read_csv(StringIO(row_str), sep='^', header=None, quoting=1)  # Use StringIO from io module
        row = row_df.iloc[0].tolist()
        return row
    except pd.errors.ParserError:
        # Pandas logic failed, use delimiter counting
        delimiter_count = row_str.count('^')
        expected_count = 15  # Assuming 16 fields, thus 15 delimiters
        if delimiter_count == expected_count:
            row = [field.strip() for field in row_str.split('^')]
            return row
        else:
            exceptions.append(line_num)
            return None #skip row

with open(movie_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        row = process_row(line.strip(), line_num)
        if row:
            try:
                embedding = model.encode(row[2], convert_to_numpy=True).tolist()
                # Take all metadata
                metadata = {str(i): v for i, v in enumerate(row)}
                metadata_size = len(json.dumps(metadata).encode('utf-8'))
                record_size = VECTOR_SIZE + metadata_size
                vectorized_data.append((row[0], embedding, metadata))
                total_size += record_size
            except Exception as e:
                exceptions.append(line_num)

print(f"Vectorized data generated. Total size: {total_size / (1024 * 1024):.2f} MB")
if exceptions:
    print(f"Exceptions occurred at lines: {exceptions}")

def upload_to_pinecone(index, data, max_batch_size=100, max_request_size=2_000_000):
    batch =[]
    batch_size = 0
    vector_count = 0
    vector_size = 384 * 4
    for tconst, embedding, metadata in data:
        metadata_size = len(json.dumps(metadata).encode('utf-8'))
        record_size = vector_size + metadata_size
        if vector_count >= max_batch_size or batch_size + record_size > max_request_size:
            if batch:
                index.upsert(vectors=batch)
                batch =[]
                batch_size = 0
                vector_count = 0
        batch.append((tconst, embedding, metadata))
        batch_size += record_size
        vector_count += 1
    if batch:
        index.upsert(vectors=batch)
    print("Upload to Pinecone completed successfully!")

print("Uploading vectorized data to Pinecone...")
upload_to_pinecone(index, vectorized_data)