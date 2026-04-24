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
import json, tempfile
from datetime import datetime, timezone
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# time-zone aware timestamp
timestamp = datetime.now(timezone.utc).isoformat()

# Firebase Initialization 
if not firebase_admin._apps:
    key_dict = json.loads(st.secrets["FIREBASE_KEY"])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as f:
        json.dump(key_dict, f)
        temp_key_path = f.name
    cred = credentials.Certificate(temp_key_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Streamlit setup
st.set_page_config(page_title="E-Waste AI", page_icon="Recycle", layout="centered")

@st.cache_resource
def load_model():
    # Updated to your new model filename
    interpreter = tf.lite.Interpreter(model_path="waste_model (1).tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Exact 12-Class Mapping from your labels.json
class_mapping = {
    0: "Biological",
    1: "Brown Glass",
    2: "Cardboard",
    3: "Clothes",
    4: "E-Waste",
    5: "Green Glass",
    6: "Metal",
    7: "Paper",
    8: "Plastic",
    9: "Shoes",
    10: "Trash",
    11: "White Glass"
}

# Voice setup
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

# Streamlit UI 
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
    .general {background: #00C851; color: white;}
    .organic {background: #ffbb33; color: black;}

    .confidence {font-size: 1.5em; font-weight: bold;}
    .footer {text-align: center; color: #888; margin-top: 50px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>♻️ E-Waste Detector</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <p style='text-align:center; color:#ccc;'>Point and click. Maintain distance and good lighting</p>
    <p style='text-align:center; color:#aaa; font-size:13px;'>
    Note: MobileNetV2 works best for single-object detection. For multiple objects, YOLO may be used in future versions.
    </p>
    """,
    unsafe_allow_html=True
)

voice_on = st.checkbox("Enable Voice", value=True, key="voice")

upload_option = st.radio(
    "Choose image input method:",
    ("Use Camera", "Upload Image"),
    horizontal=True
)

if upload_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    img_file = uploaded_file
else:
    img_file = st.camera_input("Live Camera", key="camera")


if img_file:
    # Load & preprocess specifically for MobileNetV2 (float32, [-1, 1] range)
    img = Image.open(img_file).convert("RGB")
    display_img = np.array(img)
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    # Predict
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_idx = np.argmax(output)
    
    # Handle confidence based on whether TFLite kept it as float or quantized to uint8
    if output.dtype == np.uint8:
        confidence = float(np.max(output)) / 255.0
    else:
        confidence = float(np.max(output))

    specific_label = class_mapping[pred_idx]
    
    # Group the 12 classes into your existing UI CSS categories
    if specific_label == "E-Waste":
        css_class = "ewaste"
        master_category = "E-WASTE"
    elif specific_label in ["Biological", "Cardboard", "Paper"]:
        css_class = "organic"
        master_category = "ORGANIC"
    else:
        css_class = "general"
        master_category = "GENERAL"

    # Display the specific item found, but color code it by master category
    display_label = specific_label.upper()

    # Save result to Firestore (Logging specific item and its master category)
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        db.collection("detections").add({
            "timestamp": timestamp,
            "label": display_label,
            "category": master_category,
            "confidence": float(confidence),
            "method": upload_option,
        })
    except Exception as e:
        st.warning(f"Failed to log to Firestore: {e}")

    # Voice (Reads out the specific item)
    if voice_on:
        speak(f"{specific_label} detected. Confidence {int(confidence*100)} percent.")

    # Display
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(display_img, use_column_width=True)
    with col2:
        st.markdown(f"""
        <div class="result-box {css_class}">
            <h2>{display_label}</h2>
            <div class="confidence">{confidence:.1%}</div>
            <p>Confidence</p>
        </div>
        """, unsafe_allow_html=True)

    # Download
    _, buf = cv2.imencode('.jpg', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
    st.download_button("Save Photo", buf.tobytes(), f"{display_label.lower()}.jpg", "image/jpeg")

# FOOTER
st.markdown(
    """
    <p class='footer' style='text-align:center; color:#ccc;'>
    Built by Pratham | 94% Accuracy<br>
    <span style='font-size:13px; color:#aaa;'>
    This project highlights the importance of AI-driven e-waste detection for sustainable recycling.<br>
    Around <b>62 million tonnes</b> of e-waste were generated globally in 2024, with toxic metals such as lead and mercury posing serious health risks including neurological and respiratory disorders.
    </span>
    </p>
    """,
    unsafe_allow_html=True
)
