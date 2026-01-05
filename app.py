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

# ---------- GLOBAL STYLES ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #3a0ca3, #7209b7, #b5179e);
    color: white;
}

h1, h2, h3 {
    text-align: center;
    color: #f1faee;
}

.option-box {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 15px;
    padding: 30px;
    height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    border: 2px solid rgba(255,255,255,0.4);
    transition: 0.3s ease;
}

.option-box:hover {
    background: rgba(255, 255, 255, 0.30);
    transform: scale(1.05);
}

.option-title {
    font-size: 20px;
    font-weight: bold;
    margin-top: 10px;
}

.card {
    background-color: #1e1e1e;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #333;
    min-height: 150px;
}
</style>
""", unsafe_allow_html=True)

# --- 1. Load Models & Data ---
@st.cache_resource
def load_emotion_models():
    try:
        img_model = load_model('emotion_model_mobilenet_deep_convergence_final.h5')
        audio_model = load_model('audio_emotion_lstm_model.h5')
        return img_model, audio_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_music_data():
    try:
        df = pd.read_csv('music_data.csv')
    except:
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

    return music_df.iloc[sim.argsort()[-3:][::-1]]

# --- 3. User Interface ---
st.title("🎵 AI Emotion Music Player")
st.markdown("<p style='text-align:center;'>Discover music based on your mood using Visual or Audio AI</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📸 Visual Emotion", "🎤 Auditory Emotion"])

detected_emotion = None

# ---------- VISUAL TAB ----------
with tab1:
    st.subheader("How are you looking today?")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="option-box">
            📁
            <div class="option-title">Upload Image</div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    with col2:
        st.markdown("""
        <div class="option-box">
            🤳
            <div class="option-title">Take a Selfie</div>
        </div>
        """, unsafe_allow_html=True)
        selfie_image = st.camera_input("", label_visibility="collapsed")

    image_input = uploaded_image if uploaded_image else selfie_image

    if image_input:
        image = Image.open(image_input)
        st.image(image, use_container_width=True)
        if st.button("Detect Mood"):
            detected_emotion = predict_image(image)
            st.success(f"Detected Emotion: {detected_emotion}")

# ---------- AUDIO TAB ----------
with tab2:
    st.subheader("How do you sound today?")
    audio_input = st.file_uploader("Upload a 3-second voice clip (.wav)", type=['wav'])

    if audio_input:
        st.audio(audio_input)
        if st.button("Analyze Voice"):
            detected_emotion = predict_audio(audio_input)
            st.success(f"Detected Emotion: {detected_emotion}")

# --- 4. Recommendations ---
if detected_emotion:
    st.divider()
    st.subheader(f"🎧 Recommended Tracks for your {detected_emotion} Mood")

    recs = get_music_recommendation(detected_emotion)
    cols = st.columns(len(recs))

    for i, (_, row) in enumerate(recs.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div class="card">
                <h4 style="color:#1DB954;">{row['title']}</h4>
                <p style="color:#bbb;">{row['tags']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.button("▶ Play Track", key=f"play_{i}")
