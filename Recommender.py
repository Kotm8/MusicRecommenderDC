import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI(title="Music Recommender API")

track_df = pd.read_csv(r'output.csv') 
genre_df = pd.read_csv(r'genres.csv') 

user_state = {
    "liked_songs": [],
    "disliked_songs": [],
    "shown_songs": [],
    "current_recommendation": None
}
class SongAction(BaseModel):
    action: str  # "like", "dislike", or "skip"

class SongRecommendation(BaseModel):
    track_title: str
    artist_name: str
    genre_title: str
    track_url: str

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

# FastAPI Endpoints
@app.get("/recommend", response_model=SongRecommendation)
async def get_recommendation():
    """Get a new song recommendation."""
    recommendation = recommend_songs(
        user_state["liked_songs"],
        user_state["disliked_songs"],
        user_state["shown_songs"]
    )
    
    if recommendation.empty:
        raise HTTPException(status_code=404, detail="No more songs to recommend!")
    
    song = recommendation.iloc[0]
    user_state["current_recommendation"] = song.to_dict()
    user_state["shown_songs"].append(song['track_title'])
    
    return SongRecommendation(
        track_title=song['track_title'],
        artist_name=song['artist_name'],
        genre_title=song['genre_title'],
        track_url=song['track_url']
    )

@app.post("/action")
async def perform_action(action: SongAction):
    """Perform an action (like, dislike, skip) on the current recommendation."""
    if not user_state["current_recommendation"]:
        raise HTTPException(status_code=400, detail="No song currently recommended!")
    
    current_song = user_state["current_recommendation"]["track_title"]
    action_type = action.action.lower()
    
    if action_type == "like":
        if current_song not in user_state["liked_songs"]:
            user_state["liked_songs"].append(current_song)
        return {"message": f"Liked: {current_song}"}
    elif action_type == "dislike":
        if current_song not in user_state["disliked_songs"]:
            user_state["disliked_songs"].append(current_song)
        return {"message": f"Disliked: {current_song}"}
    elif action_type == "skip":
        return {"message": f"Skipped: {current_song}"}
    else:
        raise HTTPException(status_code=400, detail="Invalid action! Use 'like', 'dislike', or 'skip'.")

@app.get("/state")
async def get_user_state():
    """Get the current user state (for debugging or inspection)."""
    return {
        "liked_songs": user_state["liked_songs"],
        "disliked_songs": user_state["disliked_songs"],
        "shown_songs": user_state["shown_songs"],
        "current_recommendation": user_state["current_recommendation"]
    }

@app.post("/reset")
async def reset_state():
    """Reset the user state."""
    user_state["liked_songs"] = []
    user_state["disliked_songs"] = []
    user_state["shown_songs"] = []
    user_state["current_recommendation"] = None
    return {"message": "User state reset successfully."}

# Run the FastAPI app (for local testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)