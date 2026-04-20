# app.py (Update)
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# 1. Model aur Classes Load Karein
MODEL_PATH = "lung_cancer_detector_final.keras"
model = tf.keras.models.load_model(MODEL_PATH)
classes = ["Bengin cases", "Malignant cases", "Normal cases"]
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def preprocess_image(img):
    # Denoise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Enhancement
    clahe_img = clahe.apply(gray)
    # Normalize and Expand for model
    img_model = clahe_img.astype("float32") / 255.0
    img_model = np.expand_dims(img_model, axis=(0, -1))  # (1, 224, 224, 1)
    return img_model


def get_affected_area_box(gray_img):
    # Tumors aksar CT scan mein bright spots hote hain (CLAHE ke baad)
    # Hum thresholding use kar ke bright areas isolate karenge

    # 1. Image ko normalize karein 0-255 range mein
    norm_img = cv2.normalize(gray_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 2. Adaptive thresholding use karein bright areas isolate karne ke liye
    # Note: Lungs black hote hain, is liye hum white areas (nodules) dhund rahe hain
    thresh = cv2.adaptiveThreshold(norm_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 3. Contours (shapes) dhundhein
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 4. Sab se bada contour dhundhein (usually tumor area in enhanced image)
    # medical scans aksar noisy hote hain isliye hum sirf large areas consider karenge
    relevant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]  # Area filter

    if not relevant_contours:
        return None

    largest_contour = max(relevant_contours, key=cv2.contourArea)

    # 5. Usk gird minimum bounding rectangle (box) ka andaza lagayein
    x, y, w, h = cv2.boundingRect(largest_contour)

    return (x, y, x + w, y + h)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_bytes = file.read()

    try:
        # Decode original image for output display
        nparr = np.frombuffer(img_bytes, np.uint8)
        raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h_orig, w_orig, _ = raw_img.shape

        # Preprocess for model prediction and feature extraction
        processed_img_model = preprocess_image(cv2.resize(raw_img, (224, 224)))

        # Prediction
        preds = model.predict(processed_img_model)
        idx = np.argmax(preds)
        confidence = float(np.max(preds))

        output_img = raw_img.copy()
        pred_class = classes[idx]

        # Detection Box Logic (sirf Malignant aur Benign ke liye)
        if pred_class != "Normal cases":
            # Phir se preprocess karein gray image hasil karne ke liye detection logic ke liye
            resized_gray = cv2.resize(raw_img, (224, 224))
            resized_gray = cv2.cvtColor(resized_gray, cv2.COLOR_BGR2GRAY)
            enhanced_gray = clahe.apply(resized_gray)

            # Effected area ke gird box coordinates hasil karein (224x224 coordinates)
            box_coords = get_affected_area_box(enhanced_gray)

            if box_coords:
                x1_s, y1_s, x2_s, y2_s = box_coords

                # Scale coordinates back to original image size
                x1 = int(x1_s * w_orig / 224)
                y1 = int(y1_s * h_orig / 224)
                x2 = int(x2_s * w_orig / 224)
                y2 = int(y2_s * h_orig / 224)

                # Draw dynamic box (Red)
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Label box
                cv2.putText(output_img, "Effected Area", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # Agar thresholding failed, fallback to center box (jaisa image_3 me tha, par label specific)
                # cv2.rectangle(output_img, (int(w_orig*0.3), int(h_orig*0.3)), (int(w_orig*0.7), int(h_orig*0.7)), (0, 0, 255), 2)
                pass

        # Image ko Base64 mein convert karein taake HTML mein dikha sakein
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
    # Local server run karein
    app.run(host='0.0.0.0', port=5000, debug=True)