---
title: dental-ai-preprocessing-mobnet
emoji: 🦷
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Dental AI Preprocessing Microservice

This microservice acts as a gatekeeper for intraoral dental images. It validates image quality (blur, brightness) and classifies the dental view angle using an ONNX MobileNetV3 Small model.

## Endpoints

### `GET /health`
Returns `{"status": "healthy"}` if the server and the ONNX MobileNetV3 Small model are loaded and ready.

### `POST /analyze-view/`
Accepts a multipart form with:
- `file`: The intraoral image file.
- `expected_view`: The display name of the expected view (e.g. `Upper Occlusal View`).

Returns:
- **Match Case**: `{"match": "Yes", "processed_image": "data:image/png;base64,..."}`
- **Rejection/Mismatch Case**: `{"match": "No", "processed_image": null}`
