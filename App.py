
!wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Your complete abusive words list (as per your original message)
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

# Number of abusive samples
num_abusive = len(abusive_texts)

# Now, add non-abusive words of your choice, roughly equal in count
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

# If non-abusive list is less than abusive, repeat some non-abusive samples until balanced
while len(non_abusive_texts) < num_abusive:
    non_abusive_texts.extend(non_abusive_texts[:num_abusive - len(non_abusive_texts)])

non_abusive_texts = non_abusive_texts[:num_abusive]  # trim to exact length

# Combine texts and labels
texts = non_abusive_texts + abusive_texts
labels = [0]*num_abusive + [1]*num_abusive

# Create DataFrame
df = pd.DataFrame({'text': texts, 'label': labels})

# Vectorize texts
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(df['text'])

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_vectorized, df['label'])

# Save model and vectorizer
os.makedirs('models', exist_ok=True)
with open('models/text_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/text_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print(f" Model trained on {len(df)} samples (balanced: {num_abusive} abusive + {num_abusive} non-abusive)")
print("Model and vectorizer saved in 'models/' folder.")

# Optional: Test
test_samples = ['beautiful', 'hate', 'amazing', 'rape', 'friendly', 'stupid']
test_vec = vectorizer.transform(test_samples)
print("Test Predictions (0=Non-abusive, 1=Abusive):", model.predict(test_vec))


!pip install streamlit NumPy librosa deepface
import streamlit as st
# --- Footer ---
st.markdown("""
    <hr>
    <div style='text-align:center;'>
        <p>Developed by AI for Safety | Streamlit App</p>
    </div>
""", unsafe_allow_html=True)


# Define the Streamlit app file name
app_file = "app.py" # Ensure your Streamlit code is saved as 'app.py'

# Create the app.py file with the Streamlit code if it doesn't exist
# This is crucial because the subprocess command needs an actual file to run
streamlit_code = """
import streamlit as st
import numpy as np
import librosa
from deepface import DeepFace
import pickle
import os
import tempfile

# Ensure required directories exist if needed by your models
if not os.path.exists('models'):
    os.makedirs('models')


# --- App Config ---
st.set_page_config(page_title="Abuse Detection System", layout="centered")
st.markdown(\"\"\"
    <h1 style='text-align: center; color: #8A2BE2;'>AI-Based Abuse & Domestic Violence Detection</h1>
    <hr style='border: 2px solid #8A2BE2;'>
\"\"\", unsafe_allow_html=True)

# --- Load Models ---
model_dir = 'models'
try:
    text_model = pickle.load(open(os.path.join(model_dir, 'text_model.pkl'), 'rb'))
    text_vectorizer = pickle.load(open(os.path.join(model_dir, 'text_vectorizer.pkl'), 'rb'))
except FileNotFoundError:
    st.error("Error: Model files not found. Please upload 'text_model.pkl' and 'text_vectorizer.pkl' inside the 'models' folder.")
    st.stop()

# --- Option Selection ---
option = st.radio("Choose input type:", ['Text', 'Audio', 'Image'], horizontal=True)

# --- Text Input ---
if option == 'Text':
    st.subheader("Enter text for analysis")
    user_text = st.text_area("Type here...")
    if st.button("Detect Abuse"):
        if user_text.strip():
            vec = text_vectorizer.transform([user_text])
            prediction = text_model.predict(vec)[0]
            label = "Abusive" if prediction == 1 else "Not Abusive"
            color = "#FF6347" if prediction == 1 else "#32CD32"
            st.markdown(f"<h3 style='color:{color};'>Text Abuse Detection: {label}</h3>", unsafe_allow_html=True)
        else:
            st.warning("Please enter some text.")

# --- Audio Input ---
elif option == 'Audio':
    st.subheader("Upload Audio File (.wav)")
    audio_file = st.file_uploader("Choose file", type=["wav"])
    if audio_file:
        try:
            y, sr = librosa.load(audio_file, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            # Note: A single mean_mfcc might not be a robust indicator of abuse
            # Consider more advanced audio features or a trained model
            mean_mfcc = np.mean(mfcc)
            # This threshold (-20) is arbitrary and likely needs to be adjusted
            result = 'Verbal Abuse' if mean_mfcc < -20 else 'No Abuse'
            color = "#FF6347" if result == 'Verbal Abuse' else "#32CD32"
            st.markdown(f"<h3 style='color:{color};'>Audio Abuse Detection: {result}</h3>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing audio: {e}")

# --- Image Input ---
elif option == 'Image':
    st.subheader("Upload Face Image (jpg/png)")
    image_file = st.file_uploader("Choose image", type=["jpg", "png"])
    if image_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + image_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(image_file.getvalue())
            tmp_file_path = tmp_file.name
        try:
            # DeepFace detection can sometimes fail if no face is found
            try:
                result = DeepFace.analyze(img_path=tmp_file_path, actions=['emotion'], enforce_detection=True)[0] # enforce_detection=True is generally better for face analysis
                emotion = result.get("dominant_emotion", "")
                # Mapping emotions to abuse is a simplification; a dedicated model would be better
                abuse = 'Physical Abuse' if emotion in ['fear', 'angry', 'sad'] else 'No Abuse'
                color = "#FF6347" if abuse == 'Physical Abuse' else "#32CD32"
                st.image(image_file, caption=f"Detected Emotion: {emotion}", use_column_width=True)
                st.markdown(f"<h3 style='color:{color};'>Image Abuse Detection: {abuse}</h3>", unsafe_allow_html=True)
            except ValueError as ve:
                 st.warning(f"Could not detect a face in the image: {ve}")
                 # Handle cases where no face is detected gracefully
                 st.markdown(f"<h3 style='color:#FFA500;'>Image Abuse Detection: No face detected for analysis</h3>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error analyzing image: {e}")
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)


# --- Footer ---
st.markdown(\"\"\"
    <hr>
    <div style='text-align:center;'>
        <p>Developed by AI for Safety | Streamlit App</p>
    </div>
\"\"\", unsafe_allow_html=True)
"""

with open(app_file, "w") as f:
    f.write(streamlit_code)

!streamlit run app.py &>/content/logs.txt &

import time
time.sleep(5)  # wait for streamlit to start

!./cloudflared tunnel --url http://localhost:8501 --no-autoupdate           
