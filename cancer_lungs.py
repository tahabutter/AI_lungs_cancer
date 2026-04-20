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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2

# ==========================================
# 1. SETTINGS & DATA LOADING
# ==========================================
dataset_path = "lung_dataset\\The IQ-OTHNCCD lung cancer dataset"
classes = ["Bengin cases", "Malignant cases", "Normal cases"]

data = []
labels = []

print("[INFO] Loading & Preprocessing Images for Transfer Learning...")
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

for category in classes:
    folder_path = os.path.join(dataset_path, category)
    if not os.path.exists(folder_path): continue
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)
        if image is None: continue

        # Preprocessing Pipeline
        image = cv2.resize(image, (224, 224))
        image = cv2.GaussianBlur(image, (3, 3), 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe_img = clahe.apply(gray)

        # Convert to RGB (3 channels) for MobileNetV2
        rgb_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)

        data.append(rgb_img)
        labels.append(category)

X = np.array(data).astype("float32") / 255.0
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

print(f"Total Images: {len(X)} | Labels: {Counter(labels)}")

# ==========================================
# 2. SPLITTING & DATA AUGMENTATION
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

# Augmenting Benign cases to handle imbalance
aug_imgs, aug_lbls = [], []
target_idx = mapping["Bengin cases"]
for img, label in zip(X_train, y_train):
    if label == target_idx:
        aug_imgs.append(cv2.flip(img, 1))
        aug_lbls.append(label)

if len(aug_imgs) > 0:
    X_train = np.concatenate((X_train, np.array(aug_imgs)), axis=0)
    y_train = np.concatenate((y_train, np.array(aug_lbls)), axis=0)

X_train, y_train = shuffle(X_train, y_train, random_state=42)

# ==========================================
# 3. MODEL ARCHITECTURE (TRANSFER LEARNING)
# ==========================================
print("\n[INFO] Building MobileNetV2 Model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pre-trained weights

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ==========================================
# 4. TRAINING
# ==========================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
]

print("\nStarting Training...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                    validation_data=(X_val, y_val), callbacks=callbacks)

# ==========================================
# 5. GENERATING LINKEDIN VISUALS
# ==========================================
print("\n[INFO] Saving Visuals and Model...")

# Plot Accuracy & Loss
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc', color='#2563eb', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Acc', color='#10b981', linewidth=2)
plt.title('Model Accuracy Growth')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='#ef4444', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', color='#f59e0b', linewidth=2)
plt.title('Model Loss Reduction')
plt.legend()
plt.savefig("linkedin_learning_curves.png", dpi=300)
plt.show()

# Confusion Matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Transfer Learning')
plt.savefig("linkedin_confusion_matrix.png", dpi=300)
plt.show()

# Classification Report
print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred, target_names=classes))

# Save Final Model
model.save("lung_cancer_mobilenet_v2.keras")
print("\n✅ Success! All tasks complete.")