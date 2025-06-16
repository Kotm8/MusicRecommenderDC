import pandas as pd
import numpy as np
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
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data
track_df = pd.read_csv(r'outputSmaller.csv')  # Replace with your tracks.csv path
genre_df = pd.read_csv(r'genres.csv')
track_embeddings = np.load(r'track_embeddings.npy')  # Load autoencoder embeddings

# Validate embeddings align with track_df
if track_embeddings.shape[0] != len(track_df):
    raise ValueError("Track embeddings and track_df have mismatched lengths!")

# User state
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

def recommend_songs(
    liked_songs: List[str],
    disliked_songs: List[str],
    shown_songs: List[str],
    num_recommendations: int = 1,
    dislike_penalty: float = 0.5
) -> pd.DataFrame:
    # Validate inputs
    if track_df.empty or track_embeddings.shape[0] != len(track_df):
        raise ValueError("Invalid track_df or embeddings")

    scores = np.zeros(len(track_df))
    song_index = {title.lower().strip(): idx for idx, title in enumerate(track_df['track_title'])}

    # Process liked songs
    for song in liked_songs:
        song_key = song.lower().strip()
        if song_key in song_index:
            idx = song_index[song_key]
            similarity = cosine_similarity([track_embeddings[idx]], track_embeddings)[0]
            scores += similarity

    # Process disliked songs
    for song in disliked_songs:
        song_key = song.lower().strip()
        if song_key in song_index:
            idx = song_index[song_key]
            similarity = cosine_similarity([track_embeddings[idx]], track_embeddings)[0]
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

        # Fallback to og:audio meta tag
        if not audio_url:
            audio_tag = soup.find('meta', property='og:audio')
            if audio_tag and audio_tag.get('content'):
                audio_url = audio_tag.get('content')

        # Extract image URL from img tag
        img_tag = soup.find('img', class_='object-cover')
        if img_tag and img_tag.get('src'):
            base_image_url = img_tag['src']
            image_url = base_image_url.replace('width=400', 'width=290').replace('height=400', 'height=290') \
                if 'width=' in base_image_url else f"{base_image_url}?width=290&height=290"

        # Fallback to og:image meta tag
        if not image_url:
            image_tag = soup.find('meta', property='og:image')
            if image_tag and image_tag.get('content'):
                base_image_url = image_tag['content']
                image_url = base_image_url.replace('width=400', 'width=290').replace('height=400', 'height=290') \
                    if 'width=' in base_image_url else f"{base_image_url}?width=290&height=290"

        return FMAResponse(audio_url=audio_url, image_url=image_url)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch FMA data: {str(e)}")

@app.get("/recommend", response_model=SongRecommendation)
async def get_recommendation():
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
    return {
        "liked_songs": user_state["liked_songs"],
        "disliked_songs": user_state["disliked_songs"],
        "shown_songs": user_state["shown_songs"],
        "current_recommendation": user_state["current_recommendation"]
    }

@app.post("/reset")
async def reset_state():
    user_state["liked_songs"] = []
    user_state["disliked_songs"] = []
    user_state["shown_songs"] = []
    user_state["current_recommendation"] = None
    return {"message": "User state reset successfully."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)