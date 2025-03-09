import os
import requests
from dotenv import load_dotenv

load_dotenv()
OMDB_API_KEY = os.getenv('OMDB_API_KEY')


def get_omdb_details_by_imdb_id(imdb_id: str, media_type: str = None) -> dict:
    """
    Fetches details from OMDb using IMDb ID.

    Args:
        imdb_id: A valid IMDb ID (e.g., tt1285016).
    Returns:
        A dictionary containing the requested details. Returns an empty
        dictionary if an error occurs.
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
    

# Example usage
if __name__ == "__main__":
    imdb_id_input = input("Enter an IMDb ID (e.g., tt1285016): ")
    movie_details = get_omdb_details_by_imdb_id(imdb_id_input)
    if movie_details:
        print("\n--- Details ---")
        print(f"Plot: {movie_details.get('plot')}")
        print(f"Poster: {movie_details.get('poster')}")
        print(f"Director: {movie_details.get('director')}")
        print(f"Writer: {movie_details.get('writer')}")
        print(f"Actors: {movie_details.get('actors')}")
        print(f"Producer: {movie_details.get('producer')}")
        print(f"IMDB Rating: {movie_details.get('imdb_rating')}")
        print(f"IMDB Votes: {movie_details.get('imdb_votes')}")
        print(f"Language: {movie_details.get('language')}")
        print("--- End of Details ---")
    else:
        print("Failed to retrieve details.")