import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

tracks_df = pd.read_csv(r'outputSmaller.csv')
genres_df = pd.read_csv(r'genres.csv')

# Handle genre_id (multi-hot encoding)
tracks_df['genre_id'] = tracks_df['genre_id'].apply(lambda x: [int(g) for g in str(x).split(',')])
mlb = MultiLabelBinarizer()
genre_features = mlb.fit_transform(tracks_df['genre_id'])

# Handle album_id (one-hot encoding)
ohe = OneHotEncoder(sparse_output=False)
album_features = ohe.fit_transform(tracks_df[['album_id']])

# Combine features
X = np.hstack([genre_features, album_features])
X = torch.tensor(X, dtype=torch.float32)
num_tracks = X.shape[0]

# Map track_id to index
track_id_to_index = {tid: idx for idx, tid in enumerate(tracks_df['track_id'])}

# Define liked/disliked tracks (use track_id from tracks.csv)
liked_tracks = [2, 3, 5]  # Replace with your track IDs
disliked_tracks = [10, 20]    # Replace with your track IDs
# Convert track_id to indices
liked_indices = [track_id_to_index[tid] for tid in liked_tracks if tid in track_id_to_index]
disliked_indices = [track_id_to_index[tid] for tid in disliked_tracks if tid in track_id_to_index]

class Autoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()  # For binary features
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

# Initialize model
input_dim = X.shape[1]
model = Autoencoder(input_dim)
criterion = nn.BCELoss()  # Better for binary features
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    reconstruction, _ = model(X)
    loss = criterion(reconstruction, X)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Get track embeddings
with torch.no_grad():
    _, track_embeddings = model(X)

# Recommendation function
def recommend_tracks(liked_tracks, disliked_tracks, embeddings, top_k=5):
    liked_embeddings = embeddings[liked_tracks]  # Shape: (num_liked, embedding_dim)
    disliked_embeddings = embeddings[disliked_tracks] if disliked_tracks else torch.tensor([])
    
    # Compute average liked embedding
    liked_center = liked_embeddings.mean(dim=0)  # Shape: (embedding_dim,)
    
    # Compute similarities to liked center
    similarities = torch.nn.functional.cosine_similarity(embeddings, liked_center.unsqueeze(0))
    
    # Penalize similarities to disliked tracks
    if len(disliked_tracks) > 0:
        disliked_center = disliked_embeddings.mean(dim=0)
        dislike_similarities = torch.nn.functional.cosine_similarity(embeddings, disliked_center.unsqueeze(0))
        similarities -= 0.5 * dislike_similarities  # Adjust penalty weight as needed
    
    # Exclude liked/disliked tracks
    exclude = set(liked_tracks + disliked_tracks)
    similarities[list(exclude)] = -float('inf')
    
    # Get top-k recommendations
    top_indices = torch.topk(similarities, top_k).indices.numpy()
    return top_indices

# Generate recommendations
recommended_tracks = recommend_tracks(liked_tracks, disliked_tracks, track_embeddings)
print("Recommended tracks:", recommended_tracks)

# Map track IDs to titles for readability
track_titles = {row['track_id']: row['track_title'] for _, row in pd.read_csv(r'outputRedacted.csv').iterrows()}
print("Recommended track titles:", [track_titles.get(idx, f"Track {idx}") for idx in recommended_tracks])

# Save embeddings
np.save(r'track_embeddings.npy', track_embeddings.numpy())