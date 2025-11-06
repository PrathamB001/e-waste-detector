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

st.set_page_config(page_title="E-Waste AI", page_icon="Recycle", layout="centered")


@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="ewaste.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


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
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .ewaste {background: #ff4444; color: white;}
    .non-ewaste {background: #00C851; color: white;}
    .confidence {font-size: 1.5em; font-weight: bold;}
    .footer {text-align: center; color: #888; margin-top: 50px;}
    [data-testid="column"] {display: flex; align-items: center;}

    
    [data-testid="stVerticalBlock"] > div[data-testid="column"]:nth-child(1),
    [data-testid="stVerticalBlock"] > div[data-testid="column"]:nth-child(2) {
        display: flex;
        align-items: center;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)



st.markdown("<h1>♻️ E-Waste Detector AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#ccc;'>Point and click. Maintain distance and good lighting</p>", unsafe_allow_html=True)


voice_on = st.checkbox("Enable Voice", value=True, key="voice")


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






