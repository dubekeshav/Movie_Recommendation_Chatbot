import pandas as pd
import os
import platform

def delete_job_and_character_columns(df):
    """Deletes 'job' and 'characters' columns from a DataFrame."""
    columns_to_delete = ['job', 'characters']
    for col in columns_to_delete:
        if col in df.columns:
            del df[col]
            print(f"Deleted '{col}' column.")
        else:
            print(f"Warning: '{col}' column not found.")
    return df

def filter_categories(df):
    """Filters rows, keeping only specified categories."""
    allowed_categories = ['actor', 'actress', 'director', 'producer', 'writer']
    df = df[df['category'].isin(allowed_categories)]
    print("Filtered categories.")
    return df

def convert_to_wide_format(df):
    """Converts long form data to wide form."""
    wide_df = df.pivot_table(
        index='tconst',
        columns='category',
        values='nconst',
        aggfunc=lambda x: ','.join(x.dropna().unique())
    ).reset_index()

    wide_df.rename(columns={
        'director': 'director',
        'writer': 'writer',
        'actor': 'actor',
        'actress': 'actress',
        'producer': 'producer'
    }, inplace=True)
    del df
    print("Converted to wide format.")
    return wide_df

def process_principals(output_directory):
    """Processes title.principals.tsv and returns wide_principals DataFrame."""
    tsv_file = os.path.join(output_directory, 'title.principals.tsv')
    try:
        print("Reading title.principals.tsv...")
        df = pd.read_csv(tsv_file, sep='\t')

        print("Processing DataFrame...")
        df = delete_job_and_character_columns(df)
        df = filter_categories(df)
        wide_df = convert_to_wide_format(df)

        print("Principals processing complete.")
        return wide_df

    except FileNotFoundError:
        print(f"Error: File not found: {tsv_file}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def process_basics_and_ratings(output_directory):
    """Processes title.basics.tsv and title.ratings.tsv, merging and returns combined DataFrame."""
    
    basics_path = os.path.join(output_directory, "title.basics.tsv")
    ratings_path = os.path.join(output_directory, "title.ratings.tsv")

    try:
        print("Reading title.basics.tsv and title.ratings.tsv...")
        movies_df = pd.read_csv(basics_path, sep="\t", low_memory=False, dtype=str)
        ratings_df = pd.read_csv(ratings_path, sep="\t", low_memory=False, dtype=str)
        combined_df = movies_df.merge(ratings_df, on="tconst", how="left")

        for col, dtype in [("isAdult", "bool"), ("startYear", "numeric"), ("endYear", "numeric"), ("averageRating", "numeric")]:
            if col in combined_df.columns:
                if dtype == "bool":
                    combined_df[col] = combined_df[col].astype(bool, errors="ignore")
                else:
                    combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

        print("Basics and ratings processing complete.")
        return combined_df

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    output_directory = 'constant'
    wide_df = process_principals(output_directory)
    combined_df = process_basics_and_ratings(output_directory)

    if wide_df is not None and combined_df is not None:
        final_df = pd.merge(combined_df, wide_df, on='tconst', how='left')
        final_csv_path = os.path.join(output_directory, "movies_combined_wide.csv")
        final_df.to_csv(final_csv_path, index=False)
        print(f"\nFinal combined data (movies_combined_wide.csv) saved to {final_csv_path}!")
    else:
        print("Error: one or more processing steps failed.")