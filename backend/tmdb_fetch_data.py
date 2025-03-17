import requests
import pandas as pd
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Load TMDb API keys from .env file
load_dotenv()
API_KEYS = [
    os.getenv("TMDB_API_KEY_1"),
    os.getenv("TMDB_API_KEY_2"),
    os.getenv("TMDB_API_KEY_3"),  # Add more keys if needed
]
NUM_KEYS = len(API_KEYS)
BATCH_SIZE = 40  # Process in batches of 40
api_index = 0  # Keeps track of which API key to use

# Function to get movie description using the rotating API keys
def get_movie_description(imdb_id):
    global api_index
    api_key = API_KEYS[api_index]  # Use the current API key
    api_index = (api_index + 1) % NUM_KEYS  # Rotate to the next API key

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
file_path = "/Users/mohitbhoir/Git/Movie_Recommendation_Chatbot/constant/imdb_tconst_list.csv"  # Change this if needed

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit()

# Ensure 'description' column exists
if "description" not in df.columns:
    df["description"] = None  # Add missing column

# Filter rows where 'description' is missing
missing_desc = df[df["description"].isna()]

if missing_desc.empty:
    print("All descriptions are already fetched. No updates needed.")
else:
    print(f"Fetching descriptions for {len(missing_desc)} movies in batches of {BATCH_SIZE}...")

    # Process IMDb IDs in batches of 40
    for i in tqdm(range(0, len(missing_desc), BATCH_SIZE), desc="Updating Descriptions"):
        batch = missing_desc.iloc[i:i + BATCH_SIZE]

        for idx, row in batch.iterrows():
            imdb_id = row["tconst"]
            
            try:
                desc = get_movie_description(imdb_id)
                if desc:
                    df.at[idx, "description"] = desc  # Update only missing descriptions

            except Exception as e:
                print(f"Error fetching description for {imdb_id}: {e}")

        # Save progress after every batch
        df.to_csv(file_path, index=False)
        print(f"Batch {i // BATCH_SIZE + 1} saved. Waiting to avoid rate limits...")

        # TMDb API rate limit (40 requests per 10 seconds) → We use 3 keys, so wait is reduced
        #time.sleep(1)  # Shorter delay since we rotate keys

    print(f"All missing descriptions updated and saved to {file_path}.")