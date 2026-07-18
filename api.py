from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import onnxruntime as ort
from quality_check import check_image_quality
import os

# --- 1. SETUP ---
app = FastAPI(title="Dental AI ONNX Stateless API")

# Enable CORS for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Resolve the ONNX model path dynamically
MODEL_PATH = None
for name in ['mobnet_v2.onnx', 'mobilenetv3_small_100.onnx']:
    path = os.path.join(BASE_DIR, name)
    if os.path.exists(path):
        MODEL_PATH = path
        break
        
if MODEL_PATH is None:
    # Fallback to scan directory for any .onnx file
    try:
        onnx_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.onnx')]
        if onnx_files:
            MODEL_PATH = os.path.join(BASE_DIR, onnx_files[0])
    except Exception:
        pass

if MODEL_PATH is None:
    MODEL_PATH = os.path.join(BASE_DIR, 'mobnet_v2.onnx')  # ultimate default

API_CLASSES = [
    'Lower Front View', 'Lower Left View', 'Lower Occlusal View', 'Lower Right View',
    'Upper Front View', 'Upper Left View', 'Upper Occlusal View', 'Upper Right View'
]

print("Loading MobileNetV3 Small ONNX Model into memory...")
try:
    ort_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print("ONNX Model Ready (Stateless Mode)")
except Exception as e:
    print(f"Failed to load AI Model: {e}")
    ort_session = None

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

# --- 3. ENDPOINTS ---
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Dental AI ONNX Preprocessing Microservice",
        "model_loaded": ort_session is not None
    }

@app.get("/health")
async def health():
    if ort_session is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "Model not loaded"}
        )
    return {"status": "healthy"}

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
        
        if image is None or ort_session is None:
            return {"match": "No", "processed_image": None}
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # B. Quality Check (Pass the original BGR image to match cv2.COLOR_BGR2GRAY)
        is_valid, _, _ = check_image_quality(image)
        if not is_valid:
            return {"match": "No", "processed_image": None}

        # C. AI Prediction (ONNX Runtime)
        img_resized = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_normalized - mean) / std
        
        # HWC to BCHW
        img_tensor = np.transpose(img_normalized, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)

        # Run model inference
        ort_inputs = {ort_session.get_inputs()[0].name: img_tensor}
        ort_outs = ort_session.run(None, ort_inputs)
        outputs = ort_outs[0]
        
        # Get predicted indices sorted by probability descending
        sorted_indices = np.argsort(outputs[0])[::-1]
        top_2_indices = sorted_indices[:2]
        top_2_classes = [API_CLASSES[idx] for idx in top_2_indices]

        # D. Match Logic (Sanitized)
        clean_expected = expected_view.strip().lower()
        clean_top_2 = [cls.strip().lower() for cls in top_2_classes]
        is_match = clean_expected in clean_top_2

        # Detailed logging for visibility on Render logs
        print(f"[AI Pipeline] Expected View: '{expected_view}' | Top-1: '{top_2_classes[0]}' | Top-2: '{top_2_classes[1]}' | Match: {is_match}")

        # E. THE RESPONSE LOGIC
        if is_match:
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
            print(f"[AI Pipeline] Rejecting image. Expected '{expected_view}' not found in top predictions {top_2_classes}")
            return {"match": "No", "processed_image": None}

    except Exception as e:
        import traceback
        print("[AI Pipeline] Exception occurred during image validation:")
        traceback.print_exc()
        return {"match": "No", "processed_image": None}
