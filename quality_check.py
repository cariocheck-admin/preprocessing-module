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
