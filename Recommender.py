import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random


track_df = pd.read_csv(r'C:\Users\kotkh\Desktop\fma_metadata\output.csv') 
genre_df = pd.read_csv(r'C:\Users\kotkh\Desktop\fma_metadata\genres.csv') 

def get_related_genres(genre_id, genre_df):
    related = {genre_id}
    genre_row = genre_df[genre_df['genre_id'] == genre_id]
    if not genre_row.empty:
        parent_id = genre_row.iloc[0]['parent']
        top_level_id = genre_row.iloc[0]['top_level']
        if parent_id != 0:
            related.add(parent_id)
        related.add(top_level_id)
    return related

# Create feature matrix
def create_feature_matrix(tracks, genres):
    all_genres = []
    for gid in tracks['genre_id']:
        related = get_related_genres(gid, genres)
        all_genres.append(list(related))
    
    unique_genres = sorted(set(gid for sublist in all_genres for gid in sublist))
    genre_features = np.zeros((len(tracks), len(unique_genres)))
    for i, genres in enumerate(all_genres):
        for gid in genres:
            genre_features[i, unique_genres.index(gid)] = 1
    
    encoder = OneHotEncoder(sparse_output=False)
    language_features = encoder.fit_transform(tracks[['track_language_code']])
    
    scaler = StandardScaler()
    interest_features = scaler.fit_transform(tracks[['track_interest']])
    
    features = np.hstack([genre_features, language_features, interest_features])
    return features, unique_genres, encoder, scaler


def recommend_songs(liked_songs, disliked_songs, shown_songs, num_recommendations=1, dislike_penalty=0.5):
    features, _, _, _ = create_feature_matrix(track_df, genre_df)
    scores = np.zeros(len(track_df))
    
    
    for song in liked_songs:
        song_data = track_df[track_df['track_title'].str.lower() == song.lower()]
        if not song_data.empty:
            idx = song_data.index[0]
            similarity = cosine_similarity([features[idx]], features)[0]
            scores += similarity
    
    
    for song in disliked_songs:
        song_data = track_df[track_df['track_title'].str.lower() == song.lower()]
        if not song_data.empty:
            idx = song_data.index[0]
            similarity = cosine_similarity([features[idx]], features)[0]
            scores -= dislike_penalty * similarity
    
    
    exclude_indices = []
    for song in shown_songs:
        song_data = track_df[track_df['track_title'].str.lower() == song.lower()]
        if not song_data.empty:
            exclude_indices.append(song_data.index[0])
    
    valid_indices = [i for i in range(len(scores)) if i not in exclude_indices]
    if not valid_indices:
        return pd.DataFrame()
    
    
    if not liked_songs and not disliked_songs:
        random_idx = random.choice(valid_indices)
        return track_df.iloc[[random_idx]][['track_title', 'artist_name', 'genre_title', 'track_url']]
    
    
    sorted_indices = sorted(valid_indices, key=lambda i: scores[i], reverse=True)
    top_indices = sorted_indices[:num_recommendations]
    return track_df.iloc[top_indices][['track_title', 'artist_name', 'genre_title', 'track_url']]

liked_songs = []
disliked_songs = []
shown_songs = []

print("Music Recommender: Enter '1' to like, '2' to dislike, '3' to skip, or any other key to exit.")
while True:
    recommendation = recommend_songs(liked_songs, disliked_songs, shown_songs)
    if recommendation.empty:
        print("No more songs to recommend!")
        break
    
    
    song = recommendation.iloc[0]
    print(f"\nRecommended Song:")
    print(f"Title: {song['track_title']}")
    print(f"Artist: {song['artist_name']}")
    print(f"Genre: {song['genre_title']}")
    print(f"URL: {song['track_url']}")
    shown_songs.append(song['track_title'])
    
    
    choice = input("Your choice (1=Like, 2=Dislike, 3=Skip, other=Exit): ").strip()
    if choice == '1':
        liked_songs.append(song['track_title'])
        print(f"Liked: {song['track_title']}")
    elif choice == '2':
        disliked_songs.append(song['track_title'])
        print(f"Disliked: {song['track_title']}")
    elif choice == '3':
        print(f"Skipped: {song['track_title']}")
        continue  
    else:
        print("Exiting...")
        break