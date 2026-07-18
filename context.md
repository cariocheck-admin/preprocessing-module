# CarioCheck Image Validation Microservice — Replication Context

This document contains the complete context, requirements, and full source code needed to replicate the CarioCheck image-validation microservice.

The new server will run an identical service but will use a **PyTorch Swin Transformer Tiny** model (`.pth` checkpoint, ~315MB) for classification instead of the legacy TensorFlow/Keras model, reducing the classes from 9 to 8 (by removing the `noise_objects` class).

---

## 1. System Overview & Architecture

This microservice acts as a gatekeeper between the CarioCheck frontend and downstream diagnostics. It:
1. Validates the incoming image quality (checking for brightness and blur using OpenCV).
2. Predicts the dental view angle of the image using a deep learning classifier.
3. Compares the predicted view against the expected view provided by the frontend.
4. Enhances matching valid images using CLAHE (Contrast Limited Adaptive Histogram Equalization) and returns the processed image as a base64 Data URL.

---

## 2. API Contract & Expected Response Format

The server exposes a single POST endpoint `/analyze-view/` which expects a multipart form request and **must return an exact response schema**.

### Request Schema (Multipart Form)
*   **`file`**: Image file (binary).
*   **`expected_view`**: String. One of the allowed views (case-insensitive).

### Response Schema (JSON)
The frontend depends on this exact shape:

#### Match Success (Quality OK, view matches expected view):
```json
{
  "match": "Yes",
  "processed_image": "data:image/png;base64,iVBORw0KGgoAAA..."
}
```

#### Match Failure / Rejection (Quality issues, wrong view, or processing error):
```json
{
  "match": "No",
  "processed_image": null
}
```
*Note: Any internal exceptions must fail silently and return the `{"match": "No", "processed_image": null}` payload to prevent frontend crashes.*

---

## 3. PyTorch Model Integration details

*   **Model Backbone**: `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k` (instantiated via `timm`).
*   **Input Resolution**: 224 x 224 pixels.
*   **Normalization (ImageNet)**:
    *   Mean: `[0.485, 0.456, 0.406]`
    *   Standard Deviation: `[0.229, 0.224, 0.225]`
*   **Orientation Warning**: DO NOT apply any horizontal/vertical flips during inference preprocessing. Doing so corrupts anatomical left-right perspective mapping.
*   **Weights Checkpoint**: The model weights are saved inside a state dict with the key `"model_state_dict"`.

### Class Mapping
The model outputs logits corresponding to 8 classes in this alphabetical index order:

| Index | Model Class Name | API / UI Expected View (Display Name) |
|---|---|---|
| 0 | `lower_front` | `Lower Front View` |
| 1 | `lower_left` | `Lower Left View` |
| 2 | `lower_occlusal` | `Lower Occlusal View` |
| 3 | `lower_right` | `Lower Right View` |
| 4 | `upper_front` | `Upper Front View` |
| 5 | `upper_left` | `Upper Left View` |
| 6 | `upper_occlusal` | `Upper Occlusal View` |
| 7 | `upper_right` | `Upper Right View` |

---

## 4. Replicated Source Code (PyTorch Swin Tiny Edition)

### File: `requirements.txt`
Dependencies required to run the PyTorch Swin Tiny microservice:
```text
fastapi
uvicorn
python-multipart
opencv-python
numpy
torch
torchvision
timm
pillow
streamlit
```

---

### File: `api.py`
The FastAPI backend server adapted for PyTorch. Place your Swin Tiny model `.pth` weight file in the same directory as this script.

```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import torch
import torchvision.transforms as transforms
import timm
from PIL import Image
from quality_check import check_image_quality

# --- 1. SETUP ---
app = FastAPI(title="Dental AI PyTorch Stateless API")

# Enable CORS for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = 'swin_tiny_model.pth'  # Rename or set accordingly
API_CLASSES = [
    'Lower Front View', 'Lower Left View', 'Lower Occlusal View', 'Lower Right View',
    'Upper Front View', 'Upper Left View', 'Upper Occlusal View', 'Upper Right View'
]

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("⏳ Loading Swin Tiny PyTorch Model into memory...")
try:
    # 1. Instantiate the timm Swin Tiny model
    model = timm.create_model(
        'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k', 
        pretrained=False, 
        num_classes=8, 
        img_size=224
    )
    
    # 2. Load the state dictionary
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    print("✅ PyTorch Model Ready (Stateless Mode)")
except Exception as e:
    print(f"❌ Failed to load AI Model: {e}")
    model = None

# Inference Preprocessing transforms (ImageNet normalization, no spatial flipping)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- 2. ENHANCEMENT LOGIC ---
def apply_enhancements(image):
    """Processes the image in memory without saving to disk."""
    img_resized = cv2.resize(image, (1024, 1024))
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

# --- 3. ENDPOINT ---
@app.post("/analyze-view/")
async def analyze_view(
    file: UploadFile = File(...), 
    expected_view: str = Form(...) 
):
    try:
        # A. Decode image from network stream
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None or model is None:
            return {"match": "No", "processed_image": None}
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # B. Quality Check (Directly from RAM)
        is_valid, _, _ = check_image_quality(image_rgb)
        if not is_valid:
            return {"match": "No", "processed_image": None}

        # C. AI Prediction (PyTorch)
        pil_img = Image.fromarray(image_rgb)
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            preds = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(preds, dim=1).item()
            
        predicted_view = API_CLASSES[predicted_idx]

        # D. Match Logic (Sanitized)
        clean_predicted = predicted_view.strip().lower()
        clean_expected = expected_view.strip().lower()

        # E. THE RESPONSE LOGIC
        if clean_predicted == clean_expected:
            # Process and return image string
            enhanced_img = apply_enhancements(image_rgb)
            enhanced_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.png', enhanced_bgr)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return {
                "match": "Yes",
                "processed_image": f"data:image/png;base64,{img_base64}"
            }
        else:
            return {"match": "No", "processed_image": None}

    except Exception as e:
        # Silently fail with "No" to keep the UI clean
        return {"match": "No", "processed_image": None}
```

---

### File: `quality_check.py`
This module analyzes the image resolution, lighting/brightness, and blur metrics. It is copied verbatim.

```python
import cv2
import numpy as np

def check_image_quality(image):
    """
    Analyzes an image for blur and lighting issues.
    Returns: (Pass/Fail Boolean, Message, Quality_Metrics_Dict)
    """
    # Convert to Grayscale for math
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # TEST 1: BLUR DETECTION
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_threshold = 5.0  # Calibrated based on 4,050 dataset images 

    if blur_score < blur_threshold:
        return False, f"Image is too blurry (Score: {blur_score:.1f})", {"blur": blur_score}

    # TEST 2: LIGHTING CHECK
    avg_brightness = np.mean(gray)
    
    if avg_brightness < 40:
        return False, "Image is too dark. Please use flash.", {"brightness": avg_brightness}
    if avg_brightness > 240:
        return False, "Image is too bright (Overexposed).", {"brightness": avg_brightness}

    return True, "Quality OK. Ready for AI.", {"blur": blur_score, "brightness": avg_brightness}
```

---

### File: `calibrate_blur.py`
Helper utility to check minimum blur thresholds over a folder of 'perfect' baseline images.

```python
import cv2
import os
import numpy as np

def calibrate_threshold(dataset_path):
    print(f"🔍 Analyzing your perfect dataset at: {dataset_path}...\n")
    
    blur_scores = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(root, file)
                
                img = cv2.imread(img_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    score = cv2.Laplacian(gray, cv2.CV_64F).var()
                    blur_scores.append(score)
    
    if not blur_scores:
        print("❌ Could not find any images. Check your folder path.")
        return

    min_score = min(blur_scores)
    avg_score = np.mean(blur_scores)
    max_score = max(blur_scores)
    
    recommended_threshold = max(5.0, min_score - 5.0) 

    print("📊 --- CALIBRATION RESULTS ---")
    print(f"Total Images Analyzed: {len(blur_scores)}")
    print(f"Sharpest Image Score: {max_score:.2f}")
    print(f"Average Image Score:  {avg_score:.2f}")
    print(f"Blurriest 'Perfect' Image Score: {min_score:.2f}")
    print("-" * 30)
    print(f"✅ RECOMMENDED ACTION:")
    print(f"Open 'quality_check.py' and change 'blur_threshold' to: {recommended_threshold:.2f}")

if __name__ == "__main__":
    # Point this to your dataset folder to recalibrate if needed
    calibrate_threshold(r"D:\dataset")
```

---

### File: `app.py`
Streamlit application for interactive testing and local visualization of the classification, validation, and enhancement pipeline.

```python
import streamlit as st
import cv2
import numpy as np
import os
import json
import time
import torch
import torchvision.transforms as transforms
import timm
from PIL import Image
from quality_check import check_image_quality

# --- CONFIGURATION ---
MODEL_PATH = 'swin_tiny_model.pth'
OUTPUT_FOLDER = 'processed_data'
CONFIDENCE_THRESHOLD = 0.50

API_CLASSES = [
    'Lower Front View', 'Lower Left View', 'Lower Occlusal View', 'Lower Right View',
    'Upper Front View', 'Upper Left View', 'Upper Occlusal View', 'Upper Right View'
]

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- LOAD AI MODEL ---
@st.cache_resource
def load_model():
    try:
        model = timm.create_model(
            'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k', 
            pretrained=False, 
            num_classes=8, 
            img_size=224
        )
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Inference preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- HELPER FUNCTIONS ---
def apply_enhancements(image):
    """Applies standard CLAHE enhancement and resizes to 1024x1024."""
    img_resized = cv2.resize(image, (1024, 1024))
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def save_data(image, view_name, patient_id):
    """Saves the cleaned image + JSON instructions for Module 2"""
    save_path = os.path.join(OUTPUT_FOLDER, patient_id)
    os.makedirs(save_path, exist_ok=True)
    
    timestamp = int(time.time())
    safe_view = view_name.replace(" ", "_")
    img_filename = f"{patient_id}_{safe_view}_{timestamp}.png"
    json_filename = f"{patient_id}_{safe_view}_{timestamp}.json"
    
    full_img_path = os.path.join(save_path, img_filename)
    cv2.imwrite(full_img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    metadata = {
        "patient_id": patient_id,
        "view_detected": view_name,
        "processing": "CLAHE + Resize",
        "medgemma_instruction": f"Analyze this {view_name} for cavities."
    }
    with open(os.path.join(save_path, json_filename), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    return full_img_path

# --- MAIN INTERFACE ---
st.set_page_config(page_title="Dental AI Gatekeeper", page_icon="🦷")
st.title("🦷 Module 1: Intelligent Pre-Processing")
st.write("Upload a raw dental image. The system will Validate, Classify, and Enhance it.")

if model is None:
    st.error(f"Could not load AI model. Please ensure '{MODEL_PATH}' is in the same folder as this script.")
    st.stop()

patient_id = st.text_input("Enter Patient ID (e.g., P-101)", "Guest_001")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. READ IMAGE
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) 
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Raw Input")

    st.markdown("### Running Analysis...")
    
    # 2. RULE-BASED QUALITY CHECK (OpenCV)
    is_valid, message, metrics = check_image_quality(original_image)
    
    if not is_valid:
        st.error(f"❌ {message}")
    else:
        st.success(f"✅ Quality Check Passed: {message}")
        
        # 3. AI CLASSIFICATION (Swin Tiny PyTorch)
        pil_img = Image.fromarray(original_image)
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            preds = torch.softmax(outputs, dim=1)
            confidence = torch.max(preds).item()
            predicted_idx = torch.argmax(preds, dim=1).item()
            
        predicted_view = API_CLASSES[predicted_idx]
        
        # 4. CONFIDENCE CHECK
        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(f"⚠️ REJECTED: Low Confidence ({confidence*100:.1f}%). Is this a clear tooth?")
            st.write(f"Best guess was: {predicted_view}")
        else:
            # 5. ENHANCEMENT & SAVING
            with st.spinner("Applying CLAHE Enhancement & Generating JSON..."):
                enhanced_img = apply_enhancements(original_image)
                saved_path = save_data(enhanced_img, predicted_view, patient_id)
            
            with col2:
                st.image(enhanced_img, caption=f"✅ Enhanced & Ready for MedGemma\n({predicted_view})")
            
            st.success(f"Files saved to: {saved_path}")
            st.json({"View": predicted_view, "Confidence": f"{confidence*100:.2f}%", "Status": "Passed to Module 2"})
```

---

## 5. Replication Workflow Instructions (For AI / Developer)

To spin up this new microservice in a new directory, run these steps:

1.  **Initialize Directory**: Create a new target directory and copy `api.py`, `quality_check.py`, `calibrate_blur.py`, and `app.py` from this context file.
2.  **Add Checkpoint**: Copy your `swin_tiny_model.pth` (~315MB) into the root of the new directory. Ensure the name in `api.py` (`MODEL_PATH`) matches your checkpoint filename.
3.  **Install Environment**:
    ```bash
    pip install fastapi uvicorn python-multipart opencv-python numpy torch torchvision timm pillow streamlit
    ```
4.  **Run Server**:
    ```bash
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    ```
5.  **Verify the replica API**:
    Send a test POST request to `http://localhost:8000/analyze-view/` with `file` (multipart file) and `expected_view` matching one of the display names (e.g. `"Upper Occlusal View"`). Confirm the response returns `"match": "Yes"` and the CLAHE enhanced base64 PNG data-uri in `"processed_image"`.
