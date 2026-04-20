import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. DATA LOADING
# ==========================================
dataset_path = "lung_dataset\\The IQ-OTHNCCD lung cancer dataset"
# Spelling corrected to 'Benign' (Make sure your folder name matches this or 'Bengin cases')
classes = ["Bengin cases", "Malignant cases", "Normal cases"]

data = []
labels = []

print("[INFO] Loading dataset...")
for category in classes:
    folder_path = os.path.join(dataset_path, category)
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {category} not found!")
        continue
    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        image = cv2.imread(img_path)
        if image is None:
            continue
        data.append(image)
        labels.append(category)

print("Total images:", len(data))
print(Counter(labels))

# ==========================================
# 2. PREPROCESSING PIPELINE
# ==========================================
processed_data = []
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

for img in data:
    resized_img = cv2.resize(img, (224, 224))
    denoised_img = cv2.GaussianBlur(resized_img, (3, 3), 0)
    gray = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)
    clahe_img = clahe.apply(gray)
    processed_data.append(clahe_img)

X = np.array(processed_data)
X = np.expand_dims(X, axis=-1)  # Add channel dimension (H, W, 1)

# Label Encoding
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("Label Mapping:", mapping)

# Normalization
X = X.astype("float32") / 255.0

# ==========================================
# 3. DATA SPLITTING (Train, Val, Test)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

# ==========================================
# 4. DATA AUGMENTATION (For Minority Class)
# ==========================================
augmented_images = []
augmented_labels = []

target_class = mapping.get("Bengin cases", mapping.get("Benign cases"))

for img, label in zip(X_train, y_train):
    if label == target_class:
        # Flip
        flip = cv2.flip(img, 1)
        augmented_images.append(flip.reshape(224, 224, 1))
        augmented_labels.append(label)

        # Rotate
        rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        augmented_images.append(rotate.reshape(224, 224, 1))
        augmented_labels.append(label)

        # Brightness (Optimized for normalized data)
        bright = np.clip(img * 1.2, 0, 1)
        augmented_images.append(bright)
        augmented_labels.append(label)

if len(augmented_images) > 0:
    X_train = np.concatenate((X_train, np.array(augmented_images)), axis=0)
    y_train = np.concatenate((y_train, np.array(augmented_labels)), axis=0)

X_train, y_train = shuffle(X_train, y_train, random_state=42)
print(f"Final Training Set Size: {len(X_train)}")

# ==========================================
# 5. CUSTOM CNN ARCHITECTURE
# ==========================================
model = models.Sequential([
    layers.Input(shape=(224, 224, 1)), # Modern way to define input
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ==========================================
# 6. TRAINING
# ==========================================
print("\n[INFO] Starting Training...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# ==========================================
# 7. EVALUATION (CONFUSION MATRIX)
# ==========================================
print("\n[INFO] Starting Evaluation...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title('Confusion Matrix: Lung Cancer Detection')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# ==========================================
# 8. PLOTTING GRAPHS
# ==========================================
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
plt.title('Accuracy Growth')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.title('Loss Reduction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# 9. SAVE MODEL
# ==========================================
model.save("lung_cancer_detector_final.keras")
print("\n✅ Success! Model saved as 'lung_cancer_detector_final.keras'")