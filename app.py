import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import librosa
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
st.set_page_config(page_title="Emotion Music Player", layout="wide", page_icon="🎵")

# --- 1. Load Models & Data ---
@st.cache_resource
def load_emotion_models():
    try:
        # Load your trained models
        img_model = load_model('emotion_model_mobilenet_deep_convergence_final.h5')
        audio_model = load_model('audio_emotion_lstm_model.h5')
        return img_model, audio_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_music_data():
    try:
        # Attempt to load your music dataset
        df = pd.read_csv('music_data.csv')
    except:
        # Fallback dataset if file is missing or corrupted
        df = pd.DataFrame({
            'title': ["Lo-fi Study Beats", "Heavy Metal Thunder", "Happy Sunshine Pop", "Deep Melancholy"],
            'tags': ["neutral calm lofi", "angry intense rock", "happy energetic upbeat", "sad lonely acoustic"]
        })
    return df

img_model, audio_model = load_emotion_models()
music_df = load_music_data()

# --- 2. Prediction Logic ---

def predict_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    if img_model:
        preds = img_model.predict(img_array)
        classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        return classes[np.argmax(preds)]
    return "Neutral"

def predict_audio(audio_file):
    y, sr = librosa.load(audio_file, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mfcc = mfcc.reshape(1, 40, 1)
    
    if audio_model:
        preds = audio_model.predict(mfcc)
        classes = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        return classes[np.argmax(preds)].capitalize()
    return "Neutral"

def get_music_recommendation(emotion):
    mapping = {
        'Happy': 'upbeat happy energetic',
        'Sad': 'melancholy sad acoustic',
        'Angry': 'rock metal intense',
        'Neutral': 'calm chill lofi',
        'Fear': 'suspense dark ambient',
        'Surprise': 'pop groovy',
        'Calm': 'meditation relax'
    }
    query = mapping.get(emotion, 'relax')
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(music_df['tags'].fillna(''))
    query_vec = tfidf.transform([query])
    sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top 3 recommendations
    return music_df.iloc[sim.argsort()[-3:][::-1]]

# --- 3. User Interface ---

st.title("🎵 AI Emotion Music Player")
st.markdown("Discover music based on your current mood using Visual or Auditory AI.")

tab1, tab2 = st.tabs(["📸 Visual Emotion", "🎤 Auditory Emotion"])

detected_emotion = None

with tab1:
    st.subheader("How are you looking today?")
    input_method = st.radio("Choose input method:", ("Upload Image", "Take a Selfie"))
    img_file = st.file_uploader("Choose a photo...", type=['jpg', 'jpeg', 'png']) if input_method == "Upload Image" else st.camera_input("Selfie")
    
    if img_file:
        image = Image.open(img_file)
        if st.button("Detect Mood"):
            detected_emotion = predict_image(image)
            st.info(f"Visual Emotion Detected: {detected_emotion}")

with tab2:
    st.subheader("How do you sound today?")
    audio_input = st.file_uploader("Upload a 3-second voice clip (.wav)", type=['wav'])
    if audio_input:
        st.audio(audio_input)
        if st.button("Analyze Voice"):
            detected_emotion = predict_audio(audio_input)
            st.info(f"Audio Emotion Detected: {detected_emotion}")

# --- 4. Recommendations ---

if detected_emotion:
    st.divider()
    st.subheader(f"🎧 Recommended Tracks for your {detected_emotion} Mood")
    recs = get_music_recommendation(detected_emotion)
    
    cols = st.columns(len(recs))
    for i, (idx, row) in enumerate(recs.iterrows()):
        with cols[i]:
            
            st.markdown(f"""
<div style="padding:20px; border-radius:10px; background-color:#1e1e1e; border: 1px solid #333; min-height: 150px;">
    <h4 style="color: #1DB954; margin-bottom: 5px;">{row['title']}</h4>
    <p style="color:#888; font-size:0.85em;">Tags: {row['tags']}</p>
</div>
""", unsafe_allow_html=True)

            if st.button(f"Play Track", key=f"btn_{idx}"):
                st.write(f"Now playing: {row['title']}...")

