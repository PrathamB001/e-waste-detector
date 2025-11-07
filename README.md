# E-Waste Detector
**Live Web App:**  
[Try it now — Point your phone camera!](https://to-e-or-not-to-e.streamlit.app)
**Binary classifier (e-waste vs non-e-waste)** trained with **MobileNetV2** on a balanced dataset of **4,800** images.

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **95%** |
| **Precision / Recall / F1** | **0.95** for both classes |
| **Model file** | `ewaste_detector.h5` (~14 MB) |

---

## The model file is **binary**

`ewaste_detector.h5` is a **saved Keras neural network** – **do NOT open it in a text editor** (you’ll see gibberish).  
Treat it like a `.zip` or `.exe` file.

**Correct way to use it:** load with TensorFlow/Keras (Python).

---

## How to use the model

```bash
# 1. Clone the repo
git clone https://github.com/PrathamB001/e-waste-detector.git
cd e-waste-detector

# 2. Install TensorFlow
pip install tensorflow

# 3. Load and test
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("ewaste_detector.h5")
print("Model loaded!")

def predict(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = model.predict(arr, verbose=0)[0][0]
    label = "E-WASTE" if prob < 0.5 else "NON E-WASTE"
    return label, prob

label, prob = predict("your-photo.jpg")
print(f"{label} ({prob:.1%})")

```

# 📦 Cloud Data Logging (Firebase Integration)

The app securely logs **every detection result** to **Firebase Firestore**, enabling **real-time tracking** and **future analysis** of model performance.

---

## 🔹 Logged Fields

Each detection entry stores the following metadata:

| Field        | Description |
|--------------|-----------|
| `timestamp`  | UTC ISO timestamp when prediction occurred |
| `label`      | Model’s classification output (`E-WASTE` / `NON E-WASTE`) |
| `confidence` | Model confidence score (`0–1`) |
| `method`     | Input type used (`Camera` or `Upload`) |

---

## 🔹 Firestore Setup

### 1. **Create Firebase Project**
- Created a Firebase project.
- Enabled **Firestore Database** in **test mode** during development.

### 2. **Generate Service Account Key**
- Generated a `.json` service account key.
- Stored securely using **Streamlit Secrets Management**.

### 3. **Initialize Firestore in Code**

```python
from firebase_admin import credentials, firestore
import json, tempfile

# Load secret key from Streamlit
key_json = json.loads(st.secrets["FIREBASE_KEY"])

# Write key to temporary file
with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
    f.write(json.dumps(key_json))
    temp_key_path = f.name

# Initialize Firebase app
cred = credentials.Certificate(temp_key_path)
firebase_admin.initialize_app(cred)
db = firestore.client()
```
4. Log Each Inference Result
from datetime import datetime, timezone

timestamp = datetime.now(timezone.utc).isoformat()

db.collection("detections").add({
    "timestamp": timestamp,
    "label": label,
    "confidence": float(conf),
    "method": upload_option
})


🔒 Security

Firebase key is never exposed in the repository.
Stored securely via Streamlit Secrets Management.
Loaded at runtime using st.secrets["FIREBASE_KEY"].


📊 Firestore Dashboard 

<img width="1426" height="656" alt="image" src="https://github.com/user-attachments/assets/c7ab3bc4-1dd2-48b0-9d99-8ab6358b121a" />

