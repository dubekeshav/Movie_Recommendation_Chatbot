import os
import platform
import requests
import gzip
import shutil
import pandas as pd

# Check operating system and GPU availability
os_name = platform.system()
gpu_available = False
gpu_backend = None

if os_name == "Windows":
    try:
        import cudf
        gpu_available = True
        gpu_backend = "cudf"
        print("Using NVIDIA GPU with cuDF on Windows.")
    except ImportError:
        print("NVIDIA GPU not found or cuDF not installed. Using CPU on Windows.")

elif os_name == "Darwin":  # macOS
    try:
        import cudf
        gpu_available = True
        gpu_backend = "cudf"
        print("Using Apple Silicon GPU with cuDF on macOS.")
    except ImportError:
        print("Apple Silicon GPU not found or cuDF not installed. Using CPU on macOS.")
else:
    print("Unsupported operating system. Using CPU.")

def download_imdb_dataset(url, filename, output_dir="constant"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    extracted_filepath = os.path.join(output_dir, filename.replace('.tsv.gz', '.tsv'))

    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Extracting {filename}...")
        with gzip.open(filepath, 'rb') as f_in:
            with open(extracted_filepath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(f"Successfully downloaded and extracted {filename}")
        os.remove(filepath)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")
    except gzip.BadGzipFile as e:
        print(f"Error extracting {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    base_url = "https://datasets.imdbws.com/"
    datasets = {
        "title.basics.tsv.gz": "Titles, genres, release years",
        "title.ratings.tsv.gz": "IMDb ratings",
        "title.principals.tsv.gz": "Principal cast/crew",
        "name.basics.tsv.gz": "Names of cast/crew"
    }
    output_directory = "constant"

    for filename, description in datasets.items():
        url = base_url + filename
        download_imdb_dataset(url, filename, output_directory)

    print(f"IMDb dataset download complete. Files saved to {output_directory}")