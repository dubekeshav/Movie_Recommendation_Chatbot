# import pandas as pd
# import csv
# from typing import List
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # --- Configuration ---
# INPUT_CSV_FILE_PATH = "constant/output_movies_copy.txt"  # Adjust if needed
# OUTPUT_CSV_FILE_PATH = "constant/processed_movies.csv"  # New output file

# def prepare_documents_for_splitting(df: pd.DataFrame) -> List[Document]:
#     """Prepares a list of Document objects from the DataFrame with structured context."""
#     docs = []
#     for index, row in df.iterrows():
#         structured_str = (
#             f"Movie: {row.get('originalTitle','')}, A-Rated: {row.get('isAdult','')}, Year: {row.get('startYear','')}, "
#             f"Duration: {row.get('runtimeMinutes','')}, Genre: {row.get('genres','')}, "
#             f"Rating: {row.get('averageRating','')}, Number of Votes: {row.get('numVotes','')}, Actor: {row.get('actor','')}, "
#             f"Actress: {row.get('actress','')}, Director: {row.get('director','')}, Producer: {row.get('producer','')}, "
#             f"Writer: {row.get('writer','')}"
#         )
#         doc = Document(page_content=structured_str)
#         docs.append(doc)
#     return docs

# def split_movie_descriptions(docs: List[Document]) -> List[Document]:
#     """Splits movie descriptions into chunks while preserving other metadata."""
#     description_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     all_splits = []
#     for doc in docs:
#         movie_data = doc.page_content.split("Description: ")[0]
#         description = doc.page_content.split("Description: ")[1]
#         description_chunks = description_splitter.split_text(description)
#         for chunk in description_chunks:
#             new_content = f"{movie_data} Description: {chunk}"
#             new_doc = Document(page_content=new_content, metadata=doc.metadata)
#             all_splits.append(new_doc)
#     return all_splits

# def process_data():
#     """Reads CSV, prepares documents, and splits descriptions."""
#     df = pd.read_csv(INPUT_CSV_FILE_PATH, sep="^", quoting=csv.QUOTE_ALL)
#     df["primaryTitle"] = df["primaryTitle"].fillna("").astype(str)

#     docs = prepare_documents_for_splitting(df)
#     all_splits = split_movie_descriptions(docs)

#     # Convert Document objects back to DataFrame for CSV export
#     processed_data = []
#     for doc in all_splits:
#         # Assuming metadata is a dictionary
#         metadata = doc.metadata
#         # Extract metadata values in a consistent order
#         metadata_values = [metadata.get(str(i), '') for i in range(len(metadata))]
#         processed_data.append([doc.page_content] + metadata_values)

#     # Create DataFrame from processed data
#     column_names = ["page_content"] + [f"metadata_{i}" for i in range(len(all_splits[0].metadata))]
#     processed_df = pd.DataFrame(processed_data, columns=column_names)

#     # Save to new CSV file
#     processed_df.to_csv(OUTPUT_CSV_FILE_PATH, index=False)
#     print(f"Processed data saved to {OUTPUT_CSV_FILE_PATH}")

# if __name__ == "__main__":
#     process_data()

import pandas as pd
import numpy as np

def process_data(titles_file_path, credits_file_path, output_file_path):
    """Processes movie titles and credits data, handling NaN and NA values."""
    titles_df = pd.read_csv(titles_file_path)
    credits_df = pd.read_csv(credits_file_path)

    merged_df = pd.merge(titles_df, credits_df, on="id", how="left")

    aggregated_actors_df = merged_df.groupby("id").agg({
        "name": lambda x: ", ".join(x.dropna()),
        "character": lambda x: ", ".join(x.dropna()),
        "role": lambda x: ", ".join(x.dropna())
    }).reset_index()

    aggregated_actors_df.rename(columns={"name": "actors", "role": "actor_roles", "character": "characters"}, inplace=True)

    final_df = pd.merge(titles_df, aggregated_actors_df, on="id", how="left")

    final_df = final_df[["title", "type", "runtime", "release_year", "genres", "actors", "characters", "actor_roles", "description"]]

    final_df["genres"] = final_df["genres"].str.replace(r'[\[\]\']', '', regex=True)
    final_df["release_year"] = final_df["release_year"].fillna(-1).astype(int)
    final_df["runtime"] = final_df["runtime"].fillna(-1).astype(int)

    # Replace NaN and NA with empty strings
    final_df = final_df.replace({np.nan: '', 'NaN': '', 'NA': '', 'N/A': ''}, regex=False)

    print(final_df.head()) #Print out the first few rows.

    final_df['page_content'] = final_df.apply(
        lambda row: (
            f"Title: {row['title']}, Type: {row['type']}, Runtime: {row['runtime']}, "
            f"Release Year: {row['release_year']}, Genre: {row['genres']}, "
            f"Actors: {row['actors']}, Characters: {row['characters']}, Actor Roles: {row['actor_roles']}, "
            f"Description: {row['description']}"
        ),
        axis=1
    )

    final_df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    titles_file_path = "../data/movies_data/titles.csv"
    credits_file_path = "../data/movies_data/credits.csv"
    output_file_path = "formatted_movies_per_movie.csv"
    process_data(titles_file_path, credits_file_path, output_file_path)