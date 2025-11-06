# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import pyttsx3
import threading
from io import BytesIO

st.set_page_config(page_title="E-Waste AI", page_icon="♻️", layout="centered")

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="ewaste.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Voice function 
def speak(text):
    def run():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
        except:
            pass  # Fails silently on mobile (no audio support)
    threading.Thread(target=run, daemon=True).start()

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #1e3c72, #2a5298); color: white;}
    .stApp {background: transparent;}
    h1 {font-family: 'Montserrat', sans-serif; text-align: center; color: #00ff88;}
    .result-box {padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0;}
    .ewaste {background: #ff4444; color: white;}
    .non-ewaste {background: #00C851; color: white;}
    .confidence {font-size: 1.5em; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>♻️ E-Waste Detector AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#ccc;'>Point your camera at waste and click.AI detects in 0.1s</p>", unsafe_allow_html=True)

img_file = st.camera_input("Live Camera", key="camera")

if img_file:
    img = Image.open(img_file).convert("RGB")
    display_img = np.array(img)
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    prob = interpreter.get_tensor(output_details[0]['index'])[0][0]

    label = "E-WASTE" if prob < 0.5 else "NON E-WASTE"
    conf = 1 - prob if prob < 0.5 else prob
    css_class = "ewaste" if prob < 0.5 else "non-ewaste"

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

    _, buf = cv2.imencode('.jpg', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
    st.download_button("Save Photo", buf.tobytes(), f"{label.lower()}.jpg", "image/jpeg")

st.markdown("<p style='text-align:center; color:#888; margin-top:50px;'>Built by Pratham | 95% Accuracy</p>", unsafe_allow_html=True)



