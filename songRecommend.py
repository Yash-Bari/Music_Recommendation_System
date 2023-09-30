import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Set Streamlit theme and layout
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load and preprocess the dataset
data = pd.read_csv('Spotify.csv')

# Drop rows with invalid 'duration_ms' values (non-numeric)
data = data[pd.to_numeric(data['duration_ms'], errors='coerce').notna()]

# Extract numerical features (excluding 'time_signature')
numerical_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness',
                      'speechiness', 'tempo', 'valence']

# Extract categorical features and perform one-hot encoding
categorical_features = ['key']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(data[numerical_features])

# Streamlit web app
st.title("🎵 Music Recommendation System")

# Sidebar with user input
st.sidebar.header("User Input")
song_name = st.sidebar.text_input("Enter a song name:", "Shape of You")
num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=20, value=10)


# Recommendation function by song name
def recommend_tracks_by_name(song_name, num_recommendations):
    # Filter tracks that contain the given song name in their title
    matching_tracks = data[data['track_name'].str.contains(song_name, case=False, na=False)]

    if matching_tracks.empty:
        st.error(f"No tracks found with the name '{song_name}' in the dataset.")
        return

    # Calculate the mean of matching tracks' feature values to represent the song
    song_features = matching_tracks[numerical_features].mean().values.reshape(1, -1)

    # Calculate cosine similarity between the song and all other tracks
    song_similarity = cosine_similarity(song_features, data[numerical_features])

    # Get the indices of most similar tracks
    recommended_track_indices = song_similarity.argsort()[0][::-1][1:num_recommendations + 1]

    return data.iloc[recommended_track_indices][['artist_name', 'track_name']]


# Display recommendations
if st.sidebar.button("Recommend"):
    recommendations = recommend_tracks_by_name(song_name, num_recommendations)
    if recommendations is not None:
        st.header("🎶 Recommended Tracks")
        st.table(recommendations)

# Custom CSS for colorful and creative interface
st.markdown(
    """
    <style>
    body {
        background-color: #F7CAC9;
    }
    .st-bc {
        background-color: #FF6B6B;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .st-cv {
        background-color: #63D471;
        color: #FFFFFF;
        font-size: 24px;
        padding: 10px;
        border-radius: 0.5rem;
        text-align: center;
    }
    .st-bs {
        background-color: #FFD700;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .st-ek {
        background-color: #6495ED;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .st-co {
        color: #FF6B6B;
    }
    </style>
    """,
    unsafe_allow_html=True
)
