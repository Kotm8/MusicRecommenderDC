import pandas as pd


input_file = r'C:\Users\kotkh\Desktop\fma_metadata\output2.csv'
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
    'track_genres'
]

# Step 3: Drop the specified columns
# Use errors='ignore' to avoid errors if a column is missing
df_modified = df.drop(columns=columns_to_remove, errors='ignore')

# Step 4: Save the modified DataFrame to a new CSV
# Replace 'output.csv' with your desired output file name
output_file = 'output.csv'
df_modified.to_csv(output_file, index=False)

# Step 5: Print the remaining columns to verify
print("Remaining columns:", df_modified.columns.tolist())