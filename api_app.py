import base64
import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 1. Transfer Learning Model aur Classes Load Karein
MODEL_PATH = "lung_cancer_mobilenet_v2.keras"
model = tf.keras.models.load_model(MODEL_PATH)
classes = ["Bengin cases", "Malignant cases", "Normal cases"]
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def preprocess_for_mobilenet(raw_img):
    """MobileNetV2 expects (224, 224, 3) input"""
    # Resize
    img = cv2.resize(raw_img, (224, 224))
    # Denoise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Grayscale for Enhancement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe_img = clahe.apply(gray)

    # CRITICAL: Convert back to 3 channels (RGB) for Transfer Learning
    img_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)

    # Normalize
    img_rgb = img_rgb.astype("float32") / 255.0
    # Add Batch Dimension (1, 224, 224, 3)
    return np.expand_dims(img_rgb, axis=0)


def get_affected_area_box(raw_img):
    """Find the most probable affected area using image processing"""
    resized = cv2.resize(raw_img, (224, 224))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)

    # Isolate bright spots (nodules)
    norm_img = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    thresh = cv2.adaptiveThreshold(norm_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    relevant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

    if not relevant_contours:
        return None

    largest_contour = max(relevant_contours, key=cv2.contourArea)
    return cv2.boundingRect(largest_contour)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_bytes = file.read()

    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h_orig, w_orig, _ = raw_img.shape

        # Preprocess using the new 3-channel logic
        processed_input = preprocess_for_mobilenet(raw_img)

        # Prediction
        preds = model.predict(processed_input)
        idx = np.argmax(preds)
        confidence = float(np.max(preds))

        pred_class = classes[idx]
        output_img = raw_img.copy()

        # Dynamic Box Logic
        if pred_class != "Normal cases":
            box = get_affected_area_box(raw_img)
            if box:
                bx, by, bw, bh = box
                # Scale to original size
                x1 = int(bx * w_orig / 224)
                y1 = int(by * h_orig / 224)
                x2 = int((bx + bw) * w_orig / 224)
                y2 = int((by + bh) * h_orig / 224)

                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(output_img, f"Potential {pred_class}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Convert result to Base64 for HTML
        _, buffer = cv2.imencode('.jpg', output_img)
        img_as_text = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'prediction': pred_class,
            'confidence': round(confidence * 100, 2),
            'image': img_as_text,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)