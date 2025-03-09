import pinecone
from pinecone import ServerlessSpec, Pinecone
import os
import time
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create an index
index_name = "movies-actors"
if index_name not in pc.list_indexes().names():  # Corrected index name check
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
    while index_name not in pc.list_indexes().names(): #wait until index is created.
        time.sleep(1)

# Connect to the index
index = pc.Index(index_name)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load actors dataset in chunks
actor_file = "./constant/name.basics.tsv"
reader = pd.read_csv(actor_file, chunksize=100000, sep='\t', low_memory=False) #added seperator and low_memory

for chunk in reader:
    vectors = []
    for _, row in chunk.iterrows():
        try:
            text_rep = f"{row['primaryName']} {row['primaryProfession']} {row['knownForTitles']}"
            vector = model.encode(text_rep).tolist()
            vectors.append((row["nconst"], vector, {"name": row["primaryName"], "knownFor": row["knownForTitles"]}))
        except KeyError:
            print(f"Skipping row due to missing keys: {row}") #Handle missing keys.

    # Upload vectors in smaller batches
    batch_size = 100  # Adjust based on your dataset size
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors[i:i + batch_size])

print("Actors data uploaded to Pinecone!")

# Load movie dataset in chunks
movie_file = "./constant/movies_combined_wide.tsv"
reader = pd.read_csv(movie_file, chunksize=100000, sep='\t', low_memory=False) #added seperator and low_memory

for chunk in reader:
    vectors = []
    for _, row in chunk.iterrows():
        try:
            text_rep = f"{row['primaryTitle']} {row['genres']} {row['actor']} {row['director']} {row['writer']}"
            vector = model.encode(text_rep).tolist()
            vectors.append((row["tconst"], vector, {"title": row["primaryTitle"], "genres": row["genres"], "director": row["director"]}))
        except KeyError:
            print(f"Skipping row due to missing keys: {row}") #Handle missing keys

    # Upload vectors in smaller batches
    batch_size = 100  # Adjust as needed
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors[i:i + batch_size])

print("Movies data uploaded to Pinecone!")

query = model.encode("Sci-fi action movie with aliens").tolist()

response = index.query(
    vector=query,
    top_k=5,  # Retrieve top 5 similar movies
    include_metadata=True
)

print(response)