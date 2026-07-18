import io
import numpy as np
import cv2
from PIL import Image
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def create_valid_dummy_image():
    # Create a 224x224 RGB image with noise/patterns to pass blur checks
    img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    # Draw some shapes/lines to add distinct edges (increases laplacian variance)
    for i in range(10):
        cv2.line(img, (np.random.randint(0, 224), np.random.randint(0, 224)),
                 (np.random.randint(0, 224), np.random.randint(0, 224)),
                 (0, 0, 0), 2)
    # Check if this dummy image passes quality check
    from quality_check import check_image_quality
    passed, msg, metrics = check_image_quality(img)
    print(f"Dummy image quality check: passed={passed}, message={msg}, metrics={metrics}")
    
    # Save to memory buffer as PNG
    _, buffer = cv2.imencode('.png', img)
    return io.BytesIO(buffer.tobytes())

def test_endpoints():
    print("Testing GET / ...")
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    print("GET / Response:", data)
    assert data["status"] == "online"
    
    print("\nTesting GET /health ...")
    response = client.get("/health")
    print("GET /health status code:", response.status_code)
    # Depending on whether model is loaded successfully:
    if response.status_code == 200:
        print("Model is successfully loaded and healthy!")
    else:
        print("Model loading failed or is pending. Details:", response.json())

    print("\nTesting POST /analyze-view/ with invalid expected view...")
    img_data = create_valid_dummy_image()
    files = {"file": ("test.png", img_data, "image/png")}
    data = {"expected_view": "Invalid View"}
    response = client.post("/analyze-view/", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    print("POST /analyze-view/ (mismatch case) response:", result)
    assert result["match"] == "No"
    assert result["processed_image"] is None

    # Let's test with all possible valid views to see what view our model predicts for this dummy image
    print("\nTesting POST /analyze-view/ to find predicted view of dummy image...")
    api_classes = [
        'Lower Front View', 'Lower Left View', 'Lower Occlusal View', 'Lower Right View',
        'Upper Front View', 'Upper Left View', 'Upper Occlusal View', 'Upper Right View'
    ]
    predicted_view = None
    for view in api_classes:
        img_data.seek(0)
        files = {"file": ("test.png", img_data, "image/png")}
        data = {"expected_view": view}
        response = client.post("/analyze-view/", files=files, data=data)
        result = response.json()
        if result["match"] == "Yes":
            predicted_view = view
            print(f"Matched successfully with expected_view='{view}'!")
            assert result["processed_image"] is not None
            assert result["processed_image"].startswith("data:image/png;base64,")
            break

    if predicted_view:
        print(f"\nSuccess: The model classified the dummy image as '{predicted_view}' and returned a processed image!")
    else:
        print("\nNote: The model did not match any of the 8 classes (or quality/model failed). check logs.")

if __name__ == "__main__":
    test_endpoints()
