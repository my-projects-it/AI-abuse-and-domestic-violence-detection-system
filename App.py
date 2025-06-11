import streamlit as st
import numpy as np
import librosa
from deepface import DeepFace
import pickle
import os
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Ensure model directory
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# --- Auto Generate Model if Not Found ---
model_path = os.path.join(model_dir, 'text_model.pkl')
vectorizer_path = os.path.join(model_dir, 'text_vectorizer.pkl')

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    abusive_texts = [
        'hate', 'loser', 'disgusting', 'stupid', 'worthless',
        'pagal', 'galiyan', 'hit', 'slap', 'punch',
        'kick', 'burn', 'choke', 'cut', 'dragged',
        'beaten', 'pushed', 'thrown', 'stabbed', 'yelled at',
        'insulted', 'blamed', 'abused', 'shouted at', 'called names',
        'humiliated', 'swore at', 'ignored', 'isolated', 'manipulated',
        'gaslighted', 'emotionally blackmailed', 'silent treatment', 'mocked', 'controlled',
        'shamed', 'pressured', 'belittled', 'made me feel worthless', 'made me cry',
        'constant criticism', 'restricted me', 'monitored me', 'not allowed to talk', 'cut off my friends',
        'overcontrolled', 'took away my money', 'no access to bank', 'financially controlled', 'stopped from working',
        'demanded my salary', 'stole from me', 'withheld money', 'forced me to depend', 'touched me inappropriately',
        'rape', 'molest', 'forced sex', 'sexual harassment', 'groped',
        'flashed', 'exposed', 'sextorted', 'non-consensual', 'coercion',
        'child abuse', 'incest', 'explicit threats', 'sexual assault', 'unwanted touch',
        'sexual comments', 'staring', 'lewd remarks', 'catcalling', 'obscene gestures',
        'showed private parts', 'sexual jokes', 'inappropriate photos', 'sexual advances', 'sending nudes',
        'online harassment', 'sexually suggestive', 'pressured for sex', 'forced to watch porn', 'inappropriate messages',
        'asked for sexual favor', 'workplace harassment', 'sexual texts', 'unwelcome attention', 'body shaming',
        'made me uncomfortable', 'beaten by husband', 'threatened by in-laws', 'abused in marriage', 'forced to stay silent',
        'not allowed to leave house', 'family abuse', 'harassed by relatives', 'domestic violence', 'controlled by partner',
        'marital rape', 'emotionally tortured at home', 'insulted by spouse', 'brainwashed', 'mentally abused',
        'psychological pressure', 'made me doubt myself', 'intimidated', 'manipulated constantly', 'felt mentally broken',
        'always afraid', 'forced to stay quiet', 'gaslighting', 'verbal attack', 'mocked daily',
        'manipulative control', 'paranoid because of them'
    ]

    non_abusive_texts = [
        'beautiful', 'good personality', 'amazing', 'nice person', 'smart and kind', 'happy', 'friendly', 'awesome', 'great job', 'well done',
        'pleasant', 'cheerful', 'joyful', 'nice work', 'excellent',
        'helpful', 'kind', 'thoughtful', 'gentle', 'polite',
        'friendly smile', 'encouraging', 'supportive', 'caring', 'respectful',
        'honest', 'brave', 'generous', 'hardworking', 'creative',
        'optimistic', 'patient', 'reliable', 'sincere', 'trustworthy',
        'understanding', 'witty', 'enthusiastic', 'cooperative', 'dedicated',
        'responsible', 'courteous', 'friendly attitude', 'pleasant manner', 'good listener',
        'team player', 'loving', 'compassionate', 'friendly neighbor', 'nice friend',
        'confident', 'resilient', 'talented', 'motivated', 'thought leader',
        'positive', 'cheerleader', 'mentor', 'inspirational', 'bright',
        'diligent', 'patient listener', 'innovative', 'strong', 'kindhearted',
        'empathetic', 'respectful colleague', 'hard worker', 'fast learner', 'adaptable',
        'dedicated worker', 'calm', 'balanced', 'honorable', 'grounded',
        'team-oriented', 'polished', 'friendly face', 'joy bringer', 'motivator',
        'open minded', 'optimistic thinker', 'energetic', 'goal-oriented', 'reliable friend',
        'clear communicator', 'organized', 'creative thinker', 'support system', 'helpful hand',
        'thoughtful giver', 'generous soul', 'warm heart', 'patient guide', 'loyal friend',
        'peaceful', 'graceful', 'cheerful companion', 'considerate', 'gentle spirit',
        'smart decision maker', 'problem solver', 'funny', 'witty', 'bright star'
    ]

    texts = non_abusive_texts[:len(abusive_texts)] + abusive_texts
    labels = [0] * len(abusive_texts) + [1] * len(abusive_texts)

    df = pd.DataFrame({'text': texts, 'label': labels})
    vec = TfidfVectorizer()
    X = vec.fit_transform(df['text'])
    model = LogisticRegression()
    model.fit(X, df['label'])

    pickle.dump(model, open(model_path, 'wb'))
    pickle.dump(vec, open(vectorizer_path, 'wb'))

# --- Load Models ---
text_model = pickle.load(open(model_path, 'rb'))
text_vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# --- App Config ---
st.set_page_config(page_title="Abuse Detection", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: violet;'>💜 AI-Based Abuse & Domestic Violence Detection 🔍</h1>
    <hr style='border: 2px solid violet;'>
""", unsafe_allow_html=True)

# --- Input Option ---
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
            y, sr = librosa.load(audio_file, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            mean_mfcc = np.mean(mfcc)
            result = 'Verbal Abuse 🔴' if mean_mfcc < -20 else 'No Abuse 🟢'
            color = "#FF6347" if 'Abuse' in result else "#32CD32"
            st.markdown(f"<h3 style='color:{color};'>Audio Abuse Detection: {result}</h3>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing audio: {e}")

# --- Image Detection ---
elif option == 'Image':
    st.subheader("🖼️ Upload Face Image (jpg/png)")
    image_file = st.file_uploader("Choose image", type=["jpg", "png"])
    if image_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + image_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(image_file.getvalue())
            tmp_file_path = tmp_file.name
        try:
            try:
                result = DeepFace.analyze(img_path=tmp_file_path, actions=['emotion'], enforce_detection=True)[0]
                emotion = result.get("dominant_emotion", "")
                abuse = 'Physical Abuse 🔴' if emotion in ['fear', 'angry', 'sad'] else 'No Abuse 🟢'
                color = "#FF6347" if 'Abuse' in abuse else "#32CD32"
                st.image(image_file, caption=f"Detected Emotion: {emotion}", use_column_width=True)
                st.markdown(f"<h3 style='color:{color};'>Image Abuse Detection: {abuse}</h3>", unsafe_allow_html=True)
            except ValueError as ve:
                st.warning(f"No face detected: {ve}")
                st.markdown(f"<h3 style='color:#FFA500;'>Image Abuse Detection: Face not detected</h3>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error analyzing image: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# --- Footer ---
st.markdown("""
    <hr>
    <div style='text-align:center;'>
        <p>🚀 Developed by AI for Safety | Streamlit App 💡</p>
    </div>
""", unsafe_allow_html=True)
