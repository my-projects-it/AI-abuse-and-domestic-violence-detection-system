💜 AI-Based Abuse & Domestic Violence Detection System 🛡️

🚺 Empowering Women. Protecting Lives. Enabling Safety.

🎯 यह AI-powered Streamlit Web App टेक्स्ट, ऑडियो और इमेज डेटा का उपयोग करके Domestic Violence और Abuse Detection करता है — instantly, intelligently और आसानी से!

🚀 Features

🔠 Text Detection:
➡️ Logistic Regression + TF-IDF के ज़रिए abusive keywords से टेक्स्ट का विश्लेषण करता है।

🔊 Audio Detection:
➡️ `librosa` से MFCC फीचर्स एक्सट्रैक्ट करके verbally abusive आवाज़ का अनुमान लगाता है।

🧠 Image (Face) Detection:
➡️ `DeepFace` मॉडल के माध्यम से facial emotion को पहचान कर possible physical abuse का संकेत देता है।

📊 Live Streamlit Interface:
➡️ Simple, clean और secure UI के साथ उपयोग करने में आसान Web App।

 🧠 Technologies Used

| Module                | Use                               |
| --------------------- | --------------------------------- |
| `scikit-learn`        | Text-based abuse classification   |
| `TF-IDF Vectorizer`   | Text vectorization                |
| `Logistic Regression` | Text classification               |
| `librosa`             | Audio feature extraction          |
| `NumPy`               | Numerical operations              |
| `DeepFace`            | Emotion analysis on facial images |
| `Streamlit`           | Web App framework                 |
| `Cloudflared`         | Public URL via ngrok replacement  |

🛠️ Installation

```bash
!pip install streamlit numpy librosa deepface
!wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
```
💼 Model Training

```python
# Text Dataset = abusive_texts + non_abusive_texts
# Train Logistic Regression on TF-IDF vectors
# Save to models/text_model.pkl and models/text_vectorizer.pkl
```

✅ Balanced dataset with 🔁 manually curated abusive & non-abusive phrases!

📦 App Structure

```
📁 models/
  ├── text_model.pkl
  └── text_vectorizer.pkl
📄 app.py
```

📌 Note: `app.py` is automatically created and launched.

🔍 Example Test Inputs

```python
test_samples = ['beautiful', 'hate', 'amazing', 'rape', 'friendly', 'stupid']
```

👁‍🗨 Prediction Output:
`0 = Non-Abusive`, `1 = Abusive`

🌐 Run the App (Google Colab Compatible)

```bash
!streamlit run app.py &>/content/logs.txt &
!./cloudflared tunnel --url http://localhost:8501 --no-autoupdate
```

🟢 Wait a few seconds and your public link will appear!

🎯 Input Options in App

* Text 📜 – Enter plain text or statements.
* Audio 🎙 – Upload `.wav` files.
* Image 🖼 – Upload face image to detect emotions.

 🧠 Emotion to Abuse Mapping (Image)

| Emotion          | Abuse Detected    |
| ---------------- | ----------------- |
| fear, angry, sad | Physical Abuse ⚠️ |
| others           | No Abuse ✅        |

 🙏 Acknowledgements

💡 Inspired by real-world need for AI-powered women & child safety tools.
📢 Use it to create awareness, build support systems, or enhance existing safety applications.

 👩‍💻 Developer

> Developed by AI for Safety ❤️
> 💬 Feel free to connect for collaboration or improvements!

🛑 Disclaimer

> ⚠️ This is a prototype and should not be used as a substitute for professional help or legal action. For emergencies, contact official helplines.
