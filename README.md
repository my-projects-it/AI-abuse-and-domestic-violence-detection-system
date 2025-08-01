# 💜 AI-Based Abuse & Domestic Violence Detection System 🛡️

🚺 **Empowering Women. Protecting Lives. Enabling Safety.**  

🎯 यह **AI-powered Streamlit Web App** टेक्स्ट, ऑडियो और इमेज डेटा का उपयोग करके  
**Domestic Violence और Abuse Detection** करता है — instantly, intelligently और आसानी से!

---

## **🚀 Features**

### 🔠 Text Abuse Detection
- **TF-IDF + Logistic Regression** पर आधारित  
- Covers **300+ curated phrases**:  
  - Physical 🥊 | Verbal 🗣 | Emotional 😢 | Financial 💰 | Mental 🧠 | Sexual ⚠️

### 🔊 Audio Abuse Detection
- **Lightweight RMS Energy Analysis**  
- Loud/shouting audio → **Verbal Abuse Alert**

### 🖼 Image (Face) Detection
- **OpenCV Haarcascade** based face detection  
- (Future GPU version will include **Emotion-based Abuse Detection**)

### 📊 Streamlit Interface
- Clean, simple and responsive UI  
- Text, Audio, Image तीनों मोड instantly switch कर सकते हैं

---

## **🧠 Technologies Used**

| Module            | Use Case                                  |
|-------------------|------------------------------------------|
| **Scikit-learn**   | Text classification (abuse / non-abuse)  |
| **TF-IDF**         | Text vectorization                       |
| **Logistic Regression** | Model for abuse detection          |
| **SoundFile + NumPy**    | Lightweight audio analysis         |
| **OpenCV**         | Face detection in uploaded images        |
| **Streamlit**      | Web app frontend                         |

---

## **📂 Project Structure**

```

abuse\_detection\_app/
│
├── app.py                # Main Streamlit App
├── models/
│   ├── text\_model.pkl     # Trained model (auto-generated if missing)
│   └── text\_vectorizer.pkl
├── requirements.txt       # Project dependencies
└── runtime.txt            # Python version (for Streamlit Cloud)

````

---

## **🛠 Installation**

1️⃣ **Clone the repository**

```bash
git clone https://github.com/my-projects-it/AI-abuse-and-domestic-violence-detection-system.git
cd AI-abuse-and-domestic-violence-detection-system
````

2️⃣ **(Optional) Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```

3️⃣ **Install Requirements**

```bash
pip install -r requirements.txt
```

4️⃣ **Run Locally**

```bash
streamlit run app.py
```

---

## **☁️ Deploy to Streamlit Cloud**

1. Push your repo to **GitHub**
2. Go to **[Streamlit Cloud](https://share.streamlit.io/)**
3. **New App → Select Repo → app.py**
4. Add `runtime.txt` for Python version:

```
python-3.10
```

## 🌐 Live Demo

🚀 **[Click Here to Open the App](https://ai-abuse-and-domestic-violence-detection-system-cvzxj7ytmhnlmv.streamlit.app/)**
*(Best viewed on Desktop for Audio & Image features)*


---

## **🔍 Detection Categories**

| Category                         | Examples                                                |
| -------------------------------- | ------------------------------------------------------- |
| **Physical Abuse** 🥊            | hit, slapped, kicked, dragged, beaten badly             |
| **Verbal Abuse** 🗣              | yelled, insulted, mocked, humiliated                    |
| **Emotional/Psychological** 😢   | gaslighted, isolated, ignored, manipulated              |
| **Sexual Harassment/Assault** ⚠️ | groped, molested, rape attempt, unwanted touch          |
| **Financial Abuse** 💰           | stole money, demanded salary, restricted bank access    |
| **Family/Domestic** 🏠           | harassed by relatives, beaten by husband, marital abuse |

---

## **📊 Example Test Inputs**

```python
test_samples = [
  "He slapped me",           # Physical Abuse
  "You are worthless",       # Verbal Abuse
  "He isolated me from all friends", # Emotional Abuse
  "She is beautiful",        # Non-Abusive
  "They demanded my salary", # Financial Abuse
  "I was molested",          # Sexual Abuse
]
```

💡 **Prediction Output:**
`0 = Non-Abusive` | `1 = Abusive`

---

## **🌐 Future Enhancements (Shakti 2.0)**

* 🔹 **Emotion-based abuse detection (DeepFace / GPU)**
* 🔹 **Real-time SOS Alerts to Police / NGOs**
* 🔹 **Heatmap Dashboard for Law Enforcement**
* 🔹 **Multi-Language Support (Hindi + Regional)**

---

## **👩‍💻 Developer**

**Barkha Jain**
💡 *AI for Safety | Tech for Social Good*

🌐 [LinkedIn](https://www.linkedin.com/in/barkha-jain-347738373) | [GitHub](https://github.com/my-projects-it)

---

## **🛑 Disclaimer**

⚠️ This prototype is **for research & awareness purposes only**.
For real emergencies, please contact **official helplines** immediately.

---

### **📜 License**

Apache-2.0 License
Use responsibly for **social good**.

```

---

Ye README **GitHub landing page jaise** sundar lagega:  
- Emoji headers  
- Proper sections  
- Professional + Social Good tone  
- Hindi-English mix jaise tumhare **first repo style** me tha  

---

Agar tum chaho to mai tumhare liye **isi README ke sath GitHub pe attractive badges aur shields bhi add** kar sakti hoon jaise **Stars, Forks, License, Tech Stack badges**.  

Kya tum chahti ho mai **badges wale premium README version** bhi bana du?
```
