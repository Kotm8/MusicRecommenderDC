import pandas as pd


input_file = r'C:\Users\kotkh\Desktop\fma_metadata\output.csv'
df = pd.read_csv(input_file)

# Step 2: Define the columns to remove
columns_to_remove = [
    'album_url',
    'artist_id',
    'artist_url',
    'artist_website',
    'license_image_file',
    'license_image_file_large',
    'license_parent_id',
    'license_title',
    'license_url',
    'track_copyright_c',
    'track_copyright_p',
    'track_date_created',
    'track_date_recorded',
    'track_disc_number',
    'tags',
    'track_comments',
    'track_composer',
    'track_explicit',
    'track_explicit_notes',
    'track_favorites',
    'track_file',
    'track_bit_rate',
    'track_information',
    'track_instrumental',
    'track_listens',
    'track_lyricist',
    'track_number',
    'track_publisher',
    'track_publisher',
    'track_genres',
    'track_image_file'
]

# Step 3: Drop the specified columns
# Use errors='ignore' to avoid errors if a column is missing
df_modified = df.drop(columns=columns_to_remove, errors='ignore')

df_modified['genre_title'] = df_modified['genre_title'].fillna('Unknown')
df_modified['track_title'] = df_modified['track_title'].fillna('Unknown Title')
df_modified['artist_name'] = df_modified['artist_name'].fillna('Unknown Artist')
df_modified['track_url'] = df_modified['track_url'].fillna('')
# Ensure track_language_code has a default value (used in feature matrix)
df_modified['track_language_code'] = df_modified['track_language_code'].fillna('unknown')
# Ensure track_interest is numeric and handle NaN (used in feature matrix)
df_modified['track_interest'] = df_modified['track_interest'].fillna(0).astype(float)
# Ensure genre_id is numeric and handle NaN
df_modified['genre_id'] = df_modified['genre_id'].fillna(0).astype(int)
# Step 4: Save the modified DataFrame to a new CSV
# Replace 'output.csv' with your desired output file name
output_file = 'output.csv'
df_modified.to_csv(output_file, index=False)

# Step 5: Print the remaining columns to verify
print("Remaining columns:", df_modified.columns.tolist())