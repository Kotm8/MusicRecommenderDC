import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import requests 
import json
from bs4 import BeautifulSoup


app = FastAPI(title="Music Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
track_df = pd.read_csv(r'outputRedacted.csv') 
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
    audio_url: Optional[str] = None
    image_url: Optional[str] = None

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

def recommend_songs(
    liked_songs: List[str],
    disliked_songs: List[str],
    shown_songs: List[str],
    num_recommendations: int = 1,
    dislike_penalty: float = 0.5
) -> pd.DataFrame:
    # Validate inputs
    features, _, _, _ = create_feature_matrix(track_df, genre_df)
    if track_df.empty or features.shape[0] != len(track_df):
        raise ValueError("Invalid track_df or feature matrix")

    scores = np.zeros(len(track_df))
    song_index = {title.lower().strip(): idx for idx, title in enumerate(track_df['track_title'])}

    # Process liked songs
    for song in liked_songs:
        song_key = song.lower().strip()
        if song_key in song_index:
            idx = song_index[song_key]
            similarity = cosine_similarity([features[idx]], features)[0]
            scores += similarity

    # Process disliked songs
    for song in disliked_songs:
        song_key = song.lower().strip()
        if song_key in song_index:
            idx = song_index[song_key]
            similarity = cosine_similarity([features[idx]], features)[0]
            scores -= dislike_penalty * similarity

    # Exclude shown songs
    exclude_indices = {song_index[song.lower().strip()] for song in shown_songs if song.lower().strip() in song_index}
    valid_indices = [i for i in range(len(scores)) if i not in exclude_indices]

    if not valid_indices:
        return pd.DataFrame(columns=['track_title', 'artist_name', 'genre_title', 'track_url', 'audio_url', 'image_url'])

    # Handle case with no preferences
    if not liked_songs and not disliked_songs:
        random_idx = random.choice(valid_indices)
        selected_row = track_df.iloc[[random_idx]]
        fma_response = extract_fma_data(selected_row['track_url'].iloc[0])
        result = selected_row[['track_title', 'artist_name', 'genre_title', 'track_url']].copy()
        result['audio_url'] = fma_response.audio_url
        result['image_url'] = fma_response.image_url
        return result

    # Select top recommendations
    sorted_indices = sorted(valid_indices, key=lambda i: scores[i], reverse=True)
    top_indices = sorted_indices[:min(num_recommendations, len(valid_indices))]
    result = track_df.iloc[top_indices][['track_title', 'artist_name', 'genre_title', 'track_url']].copy()
    result['audio_url'] = None
    result['image_url'] = None

    # Fetch FMA data for each recommended song
    for idx in result.index:
        fma_response = extract_fma_data(result.loc[idx, 'track_url'])
        result.loc[idx, 'audio_url'] = fma_response.audio_url
        result.loc[idx, 'image_url'] = fma_response.image_url

    return result

class FMAResponse(BaseModel):
    audio_url: Optional[str] = None
    image_url: Optional[str] = None
    
def extract_fma_data(track_url: str) -> FMAResponse:
    try:
        response = requests.get(track_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        audio_url = None
        image_url = None

        # Extract audio URL from data-track-info
        track_div = soup.find('div', class_='play-item')
        if track_div and 'data-track-info' in track_div.attrs:
            track_info = json.loads(track_div['data-track-info'])
            audio_url = track_info.get('fileUrl') or track_info.get('playbackUrl')

        # Fallback to og:audio meta tag if data-track-info is not found
        if not audio_url:
            audio_tag = soup.find('meta', property='og:audio')
            if audio_tag and audio_tag.get('content'):
                audio_url = audio_tag.get('content')

        # Extract image URL from img tag
        img_tag = soup.find('img', class_='object-cover')
        if img_tag and img_tag.get('src'):
            base_image_url = img_tag['src']
            # Ensure width and height parameters are set to 290
            if 'width=' in base_image_url and 'height=' in base_image_url:
                image_url = base_image_url.replace('width=400', 'width=290').replace('height=400', 'height=290')
            else:
                image_url = f"{base_image_url}&width=290&height=290" if '?' in base_image_url else f"{base_image_url}?width=290&height=290"

        # Fallback to og:image meta tag if img tag is not found
        if not image_url:
            image_tag = soup.find('meta', property='og:image')
            if image_tag and image_tag.get('content'):
                base_image_url = image_tag['content']
                if 'width=' in base_image_url and 'height=' in base_image_url:
                    image_url = base_image_url.replace('width=400', 'width=290').replace('height=400', 'height=290')
                else:
                    image_url = f"{base_image_url}&width=290&height=290" if '?' in base_image_url else f"{base_image_url}?width=290&height=290"

        return FMAResponse(audio_url=audio_url, image_url=image_url)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch FMA data: {str(e)}")

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
        track_url=song['track_url'],
        audio_url=song['audio_url'],
        image_url=song['image_url']
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