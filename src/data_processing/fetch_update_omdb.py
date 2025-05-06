import os
import requests
import pandas as pd
from dotenv import load_dotenv
import time
import csv

load_dotenv()
OMDB_API_KEY = os.getenv('OMDB_API_KEY')
INPUT_CSV_FILE_PATH = "../constant/output_movies_copy.txt"
OUTPUT_CSV_FILE_PATH = "../constant/movies_with_omdb_details.csv" 
DELAY_SECONDS = 1 #Delay seconds between API calls to avoid rate limiting.

def get_omdb_details_by_imdb_id(imdb_id: str, media_type: str = None) -> dict:
    """
    Fetches details from OMDb using IMDb ID.
    """
    if not OMDB_API_KEY:
        print("Error: OMDB_API_KEY not found in environment variables.")
        return {}

    url = "http://www.omdbapi.com/"
    params = {
        "apikey": OMDB_API_KEY,
        "i": imdb_id,
        "plot": "full",
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") == "True":
            return {
                "plot": data.get("Plot", "N/A"),
                "poster": data.get("Poster", "N/A"),
                "director": data.get("Director", "N/A"),
                "writer": data.get("Writer", "N/A"),
                "actors": data.get("Actors", "N/A"),
                "producer": data.get("Producer", "N/A"),
                "imdb_rating": data.get("imdbRating", "N/A"),
                "imdb_votes": data.get("imdbVotes", "N/A"),
                "metascore": data.get("Metascore", "N/A"),
                "language": data.get("Language", "N/A"),
            }
        else:
            print(f"OMDb API Error: {data.get('Error')}")
            return {}

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from OMDb: {e}")
        return {}
    except ValueError:
        print("Error: Invalid JSON response from OMDb.")
        return {}

def fetch_omdb_details_for_csv(input_file: str, output_file: str):
    try:
        df = pd.read_csv(input_file, sep="^", quoting=csv.QUOTE_ALL)
        df['tconst'] = df['tconst'].astype(str)
        df = df.replace(r'\\N', None, inplace=False, regex=True)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except pd.errors.ParserError:
        print(f"Error: Could not parse input file '{input_file}'.")
        return

    omdb_details_list = []
    for index, row in df.iterrows():
        imdb_id = row.get("tconst")
        if imdb_id:
            details = get_omdb_details_by_imdb_id(imdb_id)
            omdb_details_list.append(details)
            print(f"Fetched details for {imdb_id}")
            time.sleep(DELAY_SECONDS)
        else:
            omdb_details_list.append({})
            print(f"Warning: Missing IMDb ID at row {index + 1}")

    omdb_df = pd.DataFrame(omdb_details_list)
    result_df = pd.concat([df, omdb_df], axis=1)

    try:
        result_df.to_csv(output_file, index=False)
        print(f"OMDb details saved to {output_file}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

if __name__ == "__main__":
    fetch_omdb_details_for_csv(INPUT_CSV_FILE_PATH, OUTPUT_CSV_FILE_PATH)