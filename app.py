# app.py 
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import streamlit.components.v1 as components
from gtts import gTTS
import io
import base64
import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import json,tempfile
from datetime import datetime, timezone
#time-zone aware timestamp
timestamp = datetime.now(timezone.utc).isoformat()


# Firebase Initialization 
if not firebase_admin._apps:
    key_json = json.loads(st.secrets["FIREBASE_KEY"])  # decode escaped string
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
        f.write(key_json)
        temp_key_path = f.name
    cred = credentials.Certificate(temp_key_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()


#Streamlit setup
st.set_page_config(page_title="E-Waste AI", page_icon="Recycle", layout="centered")


@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="ewaste.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Voice setup
def speak(text):
    tts = gTTS(text)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_bytes = audio_fp.getvalue()
    b64_audio = base64.b64encode(audio_bytes).decode()
    components.html(f"""
    <audio autoplay>
      <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
    </audio>
    """, height=0)

#Streamlit UI 

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #1e3c72, #2a5298); color: white;}
    .stApp {background: transparent;}
    h1 {font-family: 'Montserrat', sans-serif; text-align: center; color: #00ff88; margin-bottom: 5px;}
    .subtitle {text-align: center; color: #ccc; margin-bottom: 30px;}

    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        min-height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .ewaste {background: #ff4444; color: white;}
    .non-ewaste {background: #00C851; color: white;}
    .confidence {font-size: 1.5em; font-weight: bold;}
    .footer {text-align: center; color: #888; margin-top: 50px;}

    
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        align-items: center !important; /* Center both columns vertically */
    }

    [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important; /* Centers content vertically in each col */
    }

    [data-testid="column"] > div {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }

    
    [data-testid="column"]:has(img) {
        gap: 15px !important;
        justify-content: center !important;
    }

    img {
        max-width: 100% !important;
        height: auto !important;
        display: block !important;
    }
    [data-testid="column"]:has(.result-box) .result-box {
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;

    
</style>
""", unsafe_allow_html=True)




st.markdown("<h1>♻️ E-Waste Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#ccc;'>Point and click. Maintain distance and good lighting</p>", unsafe_allow_html=True)


voice_on = st.checkbox("Enable Voice", value=True, key="voice")

#image or camera
upload_option = st.radio(
    "Choose image input method:",
    ("Use Camera", "Upload Image"),
    horizontal=True
)

#Option for image uploading 
if upload_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    img_file = uploaded_file
else:
    img_file = st.camera_input("Live Camera", key="camera")




if img_file:
    # Load & preprocess
    img = Image.open(img_file).convert("RGB")
    display_img = np.array(img)
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)

    # Predict
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    prob = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Result
    label = "E-WASTE" if prob < 0.5 else "NON E-WASTE"
    conf = 1 - prob if prob < 0.5 else prob
    css_class = "ewaste" if prob < 0.5 else "non-ewaste"

    #  Save result to Firestore 
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        db.collection("detections").add({
            "timestamp": timestamp,
            "label": label,
            "confidence": float(conf),
            "method": upload_option,
        })
    except Exception as e:
        st.warning(f"Failed to log to Firestore: {e}")


    # VOICE FIRST 
    if voice_on:
        speak(f"{label} detected. Confidence {int(conf*100)} percent.")

    # THEN DISPLAY (After voice starts) 
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(display_img, use_column_width=True)
    with col2:
        st.markdown(f"""
        <div class="result-box {css_class}">
            <h2>{label}</h2>
            <div class="confidence">{conf:.1%}</div>
            <p>Confidence</p>
        </div>
        """, unsafe_allow_html=True)

    # Download
    _, buf = cv2.imencode('.jpg', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
    st.download_button("Save Photo", buf.tobytes(), f"{label.lower()}.jpg", "image/jpeg")
 

#  FOOTER 
st.markdown("<p class='footer'>Built by Pratham | 95% Accuracy</p>", unsafe_allow_html=True)























