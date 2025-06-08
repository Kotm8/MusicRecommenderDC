import pandas as pd
import json
import numpy as np

# Load the CSV file
df = pd.read_csv(r'C:\Users\kotkh\Desktop\fma_metadata\output.csv')  # Replace with your CSV file path
def extract_genres(row):
    try:
        # Check if track_genres is a string and not NaN
        if isinstance(row['track_genres'], str):
            # Parse the track_genres string into a Python list of dictionaries
            genres = json.loads(row['track_genres'].replace("'", "\""))
            # Extract genre_id and genre_title (taking the first genre if multiple exist)
            if genres:
                return genres[0]['genre_id'], genres[0]['genre_title']
        # Return None for invalid or missing data
        return None, None
    except (json.JSONDecodeError, KeyError):
        return None, None

# Apply the function to create new columns
df[['genre_id', 'genre_title']] = df.apply(extract_genres, axis=1, result_type='expand')

# Save the updated DataFrame to a new CSV
df.to_csv('output2.csv', index=False)  # Replace with desired output file path

print("CSV updated with genre_id and genre_title columns!")