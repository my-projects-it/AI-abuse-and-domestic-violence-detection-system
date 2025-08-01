import streamlit as st
import numpy as np
import pickle
import cv2
import pandas as pd
import soundfile as sf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# --- Load Models ---
model_dir = 'models'
model_path = os.path.join(model_dir, 'text_model.pkl')
vectorizer_path = os.path.join(model_dir, 'text_vectorizer.pkl')

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    # Auto-generate lightweight text model if not found
    abusive_texts = ['hate', 'loser', 'stupid', 'worthless', 'hit', 'slap', 'abused', 'harassed', 'molest', 'rape']
    non_abusive_texts = ['beautiful', 'good', 'amazing', 'nice', 'smart', 'friendly', 'awesome', 'great job', 'well done', 'happy']

    texts = non_abusive_texts + abusive_texts
    labels = [0] * len(non_abusive_texts) + [1] * len(abusive_texts)

    df = pd.DataFrame({'text': texts, 'label': labels})
    vec = TfidfVectorizer()
    X = vec.fit_transform(df['text'])
    model = LogisticRegression()
    model.fit(X, df['label'])

    os.makedirs(model_dir, exist_ok=True)
    pickle.dump(model, open(model_path, 'wb'))
    pickle.dump(vec, open(vectorizer_path, 'wb'))

text_model = pickle.load(open(model_path, 'rb'))
text_vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# --- App Config ---
st.set_page_config(page_title="Abuse Detection App", layout="centered")
st.markdown("<h1 style='text-align:center;color:violet;'>💜 AI-Based Abuse Detection 🔍</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:2px solid violet;'>", unsafe_allow_html=True)

option = st.radio("Choose input type:", ['Text', 'Audio', 'Image'], horizontal=True)

# --- Text Detection ---
if option == 'Text':
    st.subheader("📝 Enter text for analysis")
    user_text = st.text_area("Type here...")
    if st.button("🚨 Detect Abuse"):
        if user_text.strip():
            vec = text_vectorizer.transform([user_text])
            prediction = text_model.predict(vec)[0]
            label = "Abusive 💔" if prediction == 1 else "Not Abusive 💚"
            color = "#FF6347" if prediction == 1 else "#32CD32"
            st.markdown(f"<h3 style='color:{color};'>Text Abuse Detection: {label}</h3>", unsafe_allow_html=True)
        else:
            st.warning("Please enter some text.")

# --- Audio Detection ---
elif option == 'Audio':
    st.subheader("🔊 Upload Audio File (.wav)")
    audio_file = st.file_uploader("Choose file", type=["wav"])
    if audio_file:
        try:
            data, sr = sf.read(audio_file)
            energy = np.mean(data**2)
            result = 'Verbal Abuse 🔴' if energy > 0.02 else 'No Abuse 🟢'
            color = "#FF6347" if 'Abuse' in result else "#32CD32"
            st.markdown(f"<h3 style='color:{color};'>Audio Abuse Detection: {result}</h3>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing audio: {e}")

# --- Image Detection ---
elif option == 'Image':
    st.subheader("🖼️ Upload Face Image (jpg/png)")
    image_file = st.file_uploader("Choose image", type=["jpg", "png"])
    if image_file:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            st.image(image_file, caption="Face Detected ✅", use_column_width=True)
            st.markdown("<h3 style='color:#FFA500;'>Image Abuse Detection: Face Detected (Emotion Analysis Optional)</h3>", unsafe_allow_html=True)
        else:
            st.image(image_file, caption="No Face Detected ❌", use_column_width=True)
            st.markdown("<h3 style='color:#FF6347;'>Image Abuse Detection: No Face Detected</h3>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("<hr><div style='text-align:center;'>🚀 Developed by Barkha Jain | Streamlit App 💡</div>", unsafe_allow_html=True)
