# E-Waste Detector
visit : to-e-or-not-to-e.streamlit.app
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



