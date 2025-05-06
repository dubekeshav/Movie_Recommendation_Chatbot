import requests
import pandas as pd
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from time import time, sleep

# Load TMDb API keys from .env file
load_dotenv()
API_KEYS = [
    os.getenv("TMDB_API_KEY_1"),
    os.getenv("TMDB_API_KEY_2"),
    os.getenv("TMDB_API_KEY_3"),
    os.getenv("TMDB_API_KEY_4"),  # Added the fourth API key
]
NUM_KEYS = len(API_KEYS)  # Update to reflect the new total number of keys
BATCH_SIZE = 160  # Process in batches of 160 to maximize API key usage

# Function to get movie description using the API key
def get_movie_description(imdb_id, api_key):
    url = f"https://api.themoviedb.org/3/find/{imdb_id}"
    params = {
        "api_key": api_key,
        "external_source": "imdb_id"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data["movie_results"]:
            return data["movie_results"][0].get("overview", None)
    
    return None

# Load IMDb IDs from CSV file
file_path = "/Users/mohitbhoir/Git/Movie_Recommendation_Chatbot/constant/merged_tconst.csv"  # Change this if needed

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit()

# Ensure 'description' column exists
if "description" not in df.columns:
    df["description"] = None  # Add missing column

# Filter rows where 'description' is missing and shuffle rows randomly
missing_desc = df[df["description"].isna()]

if missing_desc.empty:
    print("All descriptions are already fetched. No updates needed.")
else:
    print(f"Fetching descriptions for {len(missing_desc)} movies in batches of {BATCH_SIZE}...")

    # Process IMDb IDs in batches of 160
    with ThreadPoolExecutor(max_workers=NUM_KEYS) as executor:
        for i in tqdm(range(0, len(missing_desc), BATCH_SIZE), desc="Updating Descriptions"):
            batch = missing_desc.iloc[i:i + BATCH_SIZE]

            start_time = time()  # Track start time

            future_to_imdb = {
                executor.submit(get_movie_description, row["tconst"], API_KEYS[idx % NUM_KEYS]): idx
                for idx, row in batch.iterrows()
            }

            for future in future_to_imdb:
                idx = future_to_imdb[future]
                try:
                    desc = future.result()
                    if desc:
                        df.at[idx, "description"] = desc
                except Exception as e:
                    print(f"Error fetching description for {batch.iloc[idx]['tconst']}: {e}")

            # Save progress after every batch
            df.to_csv(file_path, index=False)
            print(f"Batch {i // BATCH_SIZE + 1} saved.")

            sleep(1)

    print(f"All missing descriptions updated and saved to {file_path}.")