import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Model load karein
model = tf.keras.models.load_model("lung_cancer_detector_final.keras")
classes = ["Bengin cases", "Malignant cases", "Normal cases"]
test_folder = "lung_dataset\\Test cases"
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

print(f"\n[INFO] Testing on unseen images from: {test_folder}")

for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)
    raw_img = cv2.imread(img_path)
    if raw_img is None: continue

    # Preprocess
    img = cv2.resize(raw_img, (224, 224))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = clahe.apply(img)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))

    # Predict
    preds = model.predict(img, verbose=0)
    idx = np.argmax(preds)
    conf = np.max(preds) * 100

    print(f"Image: {img_name} | Prediction: {classes[idx]} ({conf:.2f}%)")

    # Optional: Image show karne ke liye (Sari images ke liye window khulegi)
    plt.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    plt.title(f"{classes[idx]} - {conf:.2f}%")
    plt.axis('off')
    plt.show()