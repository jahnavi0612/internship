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
    # Loading the specific filenames for MobileNet and LSTM
    try:
        img_model = load_model('emotion_model_mobilenet_deep_convergence_final.h5')
        audio_model = load_model('audio_emotion_lstm_model.h5')
        return img_model, audio_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_music_data():
    # Mock data structure - Replace with your music_data.csv path
    try:
        df = pd.read_csv('music_data.csv')
    except:
        df = pd.DataFrame({
            'title': ["Upbeat Energy", "Midnight Blues", "Aggressive Rock", "Forest Peace", "Morning Chill", "Groovy Night"],
            'tags': ["happy energetic upbeat", "sad lonely quiet", "angry loud intense", "neutral calm relax", "calm peaceful", "surprise pop groovy"]
        })
    return df

img_model, audio_model = load_emotion_models()
music_df = load_music_data()

# --- 2. Prediction Logic ---

def predict_image(img):
    # Standard MobileNet preprocessing (224x224)
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    if img_model:
        preds = img_model.predict(img_array)
        classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        return classes[np.argmax(preds)]
    return "Neutral"

def predict_audio(audio_file):
    # Extract MFCC features for LSTM
    y, sr = librosa.load(audio_file, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mfcc = mfcc.reshape(1, 40, 1) # Match LSTM input shape (features, steps, 1)
    
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
    tfidf_matrix = tfidf.fit_transform(music_df['tags'])
    query_vec = tfidf.transform([query])
    sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    return music_df.iloc[sim.argsort()[-3:][::-1]]

# --- 3. User Interface ---

st.title("🎵 AI Emotion Music Player")
st.markdown("Discover music based on your current mood using **Visual** or **Auditory** AI.")

tab1, tab2 = st.tabs(["📸 Visual Emotion (Image/Selfie)", "🎤 Auditory Emotion (Voice)"])

detected_emotion = None

with tab1:
    st.subheader("How are you looking today?")
    
    # Selection for input method
    input_method = st.radio("Choose input method:", ("Upload Image", "Take a Selfie"))
    
    img_file = None
    if input_method == "Upload Image":
        img_file = st.file_uploader("Choose a photo...", type=['jpg', 'jpeg', 'png'])
    else:
        img_file = st.camera_input("Smile for the camera!")

    if img_file:
        image = Image.open(img_file)
        if st.button("Detect Emotion from Photo"):
            with st.spinner("Analyzing facial expressions..."):
                detected_emotion = predict_image(image)
                st.success(f"Visual Emotion Detected: **{detected_emotion}**")

with tab2:
    st.subheader("How does your voice sound?")
    audio_input = st.file_uploader("Upload a 3-second voice clip (WAV)", type=['wav'])
    
    if audio_input:
        st.audio(audio_input)
        if st.button("Detect Emotion from Voice"):
            with st.spinner("Analyzing vocal frequencies..."):
                detected_emotion = predict_audio(audio_input)
                st.success(f"Vocal Emotion Detected: **{detected_emotion}**")

# --- 4. Recommendations ---

if detected_emotion:
    st.divider()
    st.header(f"🎧 Recommended Tracks for your {detected_emotion} Mood")
    
    recs = get_music_recommendation(detected_emotion)
    
    cols = st.columns(len(recs))
    for i, (idx, row) in enumerate(recs.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div style="padding:20px; border-radius:10px; background-color:#1e1e1e; border: 1px solid #333">
                <h4>{row['title']}</h4>
                <p style="color:#888; font-size:0.8em;">{row['tags']}</p>
            </div>
            """, unsafe_渊=True)
            if st.button(f"Play Track {i+1}", key=f"play_{idx}"):
                st.toast(f"Now playing: {row['title']}")

# Instructions if nothing is detected
if not detected_emotion:
    st.info("Upload an image or record audio to get personalized music recommendations.")