import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import cv2
import soundfile as sf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="💜 AI-Based Abuse Detection", layout="centered")

st.markdown("""
    <h1 style='text-align:center;color:violet;'>💜 AI-Based Abuse & Domestic Violence Detection 🔍</h1>
    <hr style='border:2px solid violet;'>
    <p style='text-align:center;'>Text, Audio & Image-based Abuse Detection</p>
""", unsafe_allow_html=True)

# ===================== MODEL PATH ======================
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'text_model.pkl')
vectorizer_path = os.path.join(model_dir, 'text_vectorizer.pkl')

# ===================== TRAIN MODEL IF MISSING ==========
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    # --------------- ABUSIVE PHRASES (~150) ---------------
    abusive_texts = [
        # Physical Abuse
        'hit me','slapped me','kicked me','punched me','dragged me','beaten badly','choked','thrown things at me',
        'stabbed','burned','bitten','pulled my hair','locked in room','threatened to kill me','injured by partner',
        # Verbal Abuse
        'yelled at me','called me stupid','worthless','insulted me','humiliated in public','swore at me',
        'mocked daily','always shouting','called names','demeaned','sarcastic remarks','verbal attack',
        # Emotional & Psychological Abuse
        'ignored me completely','isolated me from friends','silent treatment','gaslighted me','made me doubt myself',
        'mentally tortured','controlled everything','brainwashed me','emotionally blackmailed',
        'made me feel worthless','always afraid','intimidated me','threatened to leave me','forced to stay quiet',
        'psychological pressure','felt mentally broken','manipulated constantly','paranoid because of them',
        # Sexual Harassment & Assault
        'touched me inappropriately','groped in public','harassed sexually','catcalling','sexual comments',
        'made obscene gestures','flashed private parts','molested me','rape attempt','forced sex',
        'sexual assault','marital rape','asked for sexual favors','pressured for sex','forced to watch porn',
        'sent me nudes','online harassment','sextorted','non-consensual touch','unwelcome sexual attention',
        'body shamed','staring badly','lewd remarks','sexual jokes','harassed at workplace',
        # Financial Abuse
        'took my money','stole from me','demanded my salary','financially controlled','withheld money',
        'forced me to depend financially','restricted bank access','stopped me from working',
        'forced to give money','took away property','economic abuse','stole jewelry',
        # Family / Domestic Abuse
        'beaten by husband','abused in marriage','controlled by in-laws','harassed by relatives',
        'forced to stay home','not allowed to leave house','insulted by spouse','family abuse',
        'mentally abused by family','pressured in marriage','domestic violence','always blamed by family',
        # Threats
        'explicit threats','blackmailed','life threats','threatened to harm family','used knife to scare me',
        'constant fear','feared for life','threatened by messages','intimidating calls','stalking me',
        # Extra abuse categories
        'spat on me','kicked furniture','destroyed belongings','forced confinement','ignored my health',
        'denied medical help','made me cry daily','threatened divorce','shamed my appearance','belittled my work'
    ]

    # --------------- NON ABUSIVE PHRASES (~150) ---------------
    non_abusive_texts = [
        'you are beautiful','good person','amazing friend','kind and helpful','smart and caring','awesome work',
        'great job','well done','pleasant personality','cheerful','joyful','supportive partner','caring family',
        'respectful behavior','honest and loyal','brave and strong','generous person','hardworking',
        'creative mind','optimistic approach','patient listener','reliable friend','sincere effort','trustworthy',
        'understanding','witty and fun','enthusiastic','cooperative','dedicated','responsible person',
        'courteous','loving nature','compassionate','loyal partner','confident','resilient','talented','motivated',
        'positive energy','helpful neighbor','team player','friendly attitude','calm and composed','peaceful',
        'graceful','humble','diligent','innovative thinker','good decision maker','bright student','empathetic',
        'balanced personality','grounded','open minded','energetic','goal oriented','supportive colleague',
        'organized','creative thinker','problem solver','funny and witty','loyal friend','gentle soul',
        'warm heart','considerate','respectful colleague','polished communication','optimistic thinker',
        'hard worker','fast learner','adaptable','dedicated worker','patient guide','mentor figure',
        'cheerleader','inspirational leader','bright star','thoughtful giver','generous soul','peace bringer',
        'kindhearted','team oriented','trustworthy partner','positive influence','encouraging words',
        'motivator','calm under pressure','compassionate leader','friendly neighbor','thought leader'
    ]

    # Balance lists
    texts = non_abusive_texts + abusive_texts
    labels = [0]*len(non_abusive_texts) + [1]*len(abusive_texts)

    # Train lightweight model
    df = pd.DataFrame({'text': texts, 'label': labels})
    vec = TfidfVectorizer()
    X = vec.fit_transform(df['text'])
    model = LogisticRegression()
    model.fit(X, df['label'])

    pickle.dump(model, open(model_path, 'wb'))
    pickle.dump(vec, open(vectorizer_path, 'wb'))

text_model = pickle.load(open(model_path, 'rb'))
text_vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# ===================== DETECTION FUNCTIONS ==============
def detect_text_abuse(text):
    vec = text_vectorizer.transform([text])
    pred = text_model.predict(vec)[0]
    return ("Abusive 💔","#FF6347") if pred==1 else ("Not Abusive 💚","#32CD32")

def detect_audio_abuse(file):
    try:
        data,sr=sf.read(file)
        energy=np.mean(data**2)
        return ("Verbal Abuse 🔴","#FF6347") if energy>0.02 else ("No Abuse 🟢","#32CD32")
    except: return ("Error processing audio","#FFA500")

def detect_image_face(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    return faces,img

# ===================== UI SELECTION =====================
option = st.radio("Choose input type:", ['Text','Audio','Image'], horizontal=True)

if option=='Text':
    st.subheader("📝 Enter text for analysis")
    txt=st.text_area("Type here...")
    if st.button("🚨 Detect Abuse"):
        if txt.strip():
            label,color=detect_text_abuse(txt)
            st.markdown(f"<h3 style='color:{color};'>Text Abuse Detection: {label}</h3>",unsafe_allow_html=True)
        else: st.warning("Enter some text first")

elif option=='Audio':
    st.subheader("🔊 Upload Audio (.wav)")
    file=st.file_uploader("Choose file",type=["wav"])
    if file and st.button("🚨 Detect Abuse"):
        label,color=detect_audio_abuse(file)
        st.markdown(f"<h3 style='color:{color};'>Audio Abuse Detection: {label}</h3>",unsafe_allow_html=True)

elif option=='Image':
    st.subheader("🖼️ Upload Face Image")
    file=st.file_uploader("Choose image",type=["jpg","png"])
    if file and st.button("🚨 Detect Abuse"):
        faces,img=detect_image_face(file)
        st.image(img,use_column_width=True)
        if len(faces)>0:
            st.markdown("<h3 style='color:#FFA500;'>Face Detected ✅ (Emotion Analysis Optional)</h3>",unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:#FF6347;'>No Face Detected ❌</h3>",unsafe_allow_html=True)

st.markdown("<hr><div style='text-align:center;'>🚀 Developed by Barkha Jain | AI for Safety 💡</div>",unsafe_allow_html=True)
