# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import Counter
#
# from sklearn.utils import shuffle
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
#
# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
# )
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# # ============================================================
# # 1. CONFIG
# # ============================================================
# DATASET_PATH = r"D:\lung_dataset\The IQ-OTHNCCD lung cancer dataset"
# CLASSES      = ["Bengin cases", "Malignant cases", "Normal cases"]
# IMG_SIZE     = 224
# BATCH_SIZE   = 32
# EPOCHS       = 30           # EarlyStopping will stop before this if needed
# SEED         = 42
#
# # ============================================================
# # 2. LOAD DATA
# # ============================================================
# data, labels = [], []
#
# for category in CLASSES:
#     folder_path = os.path.join(DATASET_PATH, category)
#     for img_name in os.listdir(folder_path):
#         img_path = os.path.join(folder_path, img_name)
#         image = cv2.imread(img_path)
#         if image is None:
#             print(f"Corrupted/skipped: {img_path}")
#             continue
#         data.append(image)
#         labels.append(category)
#
# print(f"Total images loaded : {len(data)}")
# print(f"Class distribution  : {Counter(labels)}")
#
# # ============================================================
# # 3. PREPROCESSING  (Resize + Gaussian Blur)
# # ============================================================
# processed = []
# for img in data:
#     resized   = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     denoised  = cv2.GaussianBlur(resized, (3, 3), 0)
#     processed.append(denoised)
#
# X = np.array(processed, dtype="float32")
#
# # ============================================================
# # 4. LABEL ENCODING
# # ============================================================
# encoder = LabelEncoder()
# y       = encoder.fit_transform(labels)
# mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
# print(f"Label mapping: {mapping}")
#
# # ============================================================
# # 5. AUGMENTATION — benign class only (most underrepresented)
# # ============================================================
# aug_images, aug_labels = [], []
#
# for img, label in zip(X, y):
#     if label == mapping["Bengin cases"]:
#         # horizontal flip
#         aug_images.append(cv2.flip(img, 1));        aug_labels.append(label)
#         # vertical flip
#         aug_images.append(cv2.flip(img, 0));        aug_labels.append(label)
#         # rotate 90
#         aug_images.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE));  aug_labels.append(label)
#         # rotate 180
#         aug_images.append(cv2.rotate(img, cv2.ROTATE_180));           aug_labels.append(label)
#         # rotate 270
#         aug_images.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)); aug_labels.append(label)
#         # brightness up
#         aug_images.append(cv2.convertScaleAbs(img, alpha=1.2, beta=20));    aug_labels.append(label)
#         # brightness down
#         aug_images.append(cv2.convertScaleAbs(img, alpha=0.8, beta=-10));   aug_labels.append(label)
#
# X = np.concatenate([X, np.array(aug_images)], axis=0)
# y = np.concatenate([y, np.array(aug_labels)], axis=0)
# print(f"After augmentation — X: {X.shape}, y: {y.shape}")
# print(f"New class distribution: {Counter(y)}")
#
# # ============================================================
# # 6. NORMALIZATION  (MobileNetV2 expects [0,1] or use preprocess_input)
# # ============================================================
# X = X / 255.0
#
# # ============================================================
# # 7. SHUFFLE + TRAIN/TEST SPLIT
# # ============================================================
# X, y = shuffle(X, y, random_state=SEED)
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=SEED,
#     stratify=y
# )
# print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
#
# # ============================================================
# # 8. CLASS WEIGHTS
# # ============================================================
# class_weights = compute_class_weight(
#     class_weight="balanced",
#     classes=np.unique(y_train),
#     y=y_train
# )
# class_weights = dict(enumerate(class_weights))
# print(f"Class weights: {class_weights}")
#
# # ============================================================
# # 9. KERAS AUGMENTATION ON TRAINING DATA (extra variety)
# # ============================================================
# train_datagen = ImageDataGenerator(
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     fill_mode="nearest"
# )
# train_gen  = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=SEED)
# steps_per_epoch = len(X_train) // BATCH_SIZE
#
# # ============================================================
# # 10. MODEL — MobileNetV2 Transfer Learning
# #     Phase 1: Train only top layers (base frozen)
# #     Phase 2: Fine-tune last 30 layers of base
# # ============================================================
#
# # --- Base model ---
# base_model = MobileNetV2(
#     input_shape=(IMG_SIZE, IMG_SIZE, 3),
#     include_top=False,
#     weights="imagenet"
# )
# base_model.trainable = False          # freeze all base layers first
#
# # --- Custom head ---
# inputs  = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# x       = base_model(inputs, training=False)
# x       = GlobalAveragePooling2D()(x)
# x       = BatchNormalization()(x)
# x       = Dense(256, activation="relu")(x)
# x       = Dropout(0.4)(x)
# x       = Dense(128, activation="relu")(x)
# x       = Dropout(0.3)(x)
# outputs = Dense(3, activation="softmax")(x)
#
# model = Model(inputs, outputs)
#
# model.compile(
#     optimizer=Adam(learning_rate=1e-3),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )
# model.summary()
#
# # ============================================================
# # 11. CALLBACKS
# # ============================================================
# early_stop = EarlyStopping(
#     monitor="val_accuracy",
#     patience=7,
#     restore_best_weights=True,
#     verbose=1
# )
# reduce_lr = ReduceLROnPlateau(
#     monitor="val_loss",
#     factor=0.3,
#     patience=3,
#     min_lr=1e-7,
#     verbose=1
# )
# checkpoint = ModelCheckpoint(
#     "best_lung_model.keras",
#     monitor="val_accuracy",
#     save_best_only=True,
#     verbose=1
# )
# callbacks = [early_stop, reduce_lr, checkpoint]
#
# # ============================================================
# # 12. PHASE 1 TRAINING — top layers only
# # ============================================================
# print("\n===== PHASE 1: Training top layers =====")
# history1 = model.fit(
#     train_gen,
#     steps_per_epoch=steps_per_epoch,
#     epochs=15,
#     validation_data=(X_test, y_test),
#     class_weight=class_weights,
#     callbacks=callbacks,
#     verbose=1
# )
#
# # ============================================================
# # 13. PHASE 2 — Fine-tune last 30 layers of MobileNetV2
# # ============================================================
# print("\n===== PHASE 2: Fine-tuning last 30 layers =====")
# base_model.trainable = True
#
# # Freeze all layers EXCEPT the last 30
# for layer in base_model.layers[:-20]:
#     layer.trainable = False
#
# # Recompile with much lower learning rate for fine-tuning
# model.compile(
#     optimizer=Adam(learning_rate=5e-6),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )
#
# history2 = model.fit(
#     train_gen,
#     steps_per_epoch=steps_per_epoch,
#     epochs=EPOCHS,
#     validation_data=(X_test, y_test),
#     class_weight=class_weights,
#     callbacks=callbacks,
#     verbose=1
# )
#
# # ============================================================
# # 14. EVALUATION
# # ============================================================
# print("\n===== FINAL EVALUATION =====")
# loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
# print(f"Test Loss     : {loss:.4f}")
# print(f"Test Accuracy : {accuracy * 100:.2f}%")
#
# y_pred  = np.argmax(model.predict(X_test), axis=1)
# classes = list(mapping.keys())
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=classes))
#
# # ============================================================
# # 15. PLOTS
# # ============================================================
#
# # Combine both phase histories
# def combine_history(h1, h2, key):
#     return h1.history[key] + h2.history[key]
#
# acc      = combine_history(history1, history2, "accuracy")
# val_acc  = combine_history(history1, history2, "val_accuracy")
# loss_val = combine_history(history1, history2, "loss")
# val_loss = combine_history(history1, history2, "val_loss")
# epochs_range = range(1, len(acc) + 1)
#
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# axes[0].plot(epochs_range, acc,     label="Train Accuracy", linewidth=2)
# axes[0].plot(epochs_range, val_acc, label="Val Accuracy",   linewidth=2, linestyle="--")
# axes[0].axvline(x=15, color="gray", linestyle=":", label="Phase 2 start")
# axes[0].set_title("Accuracy")
# axes[0].set_xlabel("Epoch")
# axes[0].set_ylabel("Accuracy")
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)
#
# axes[1].plot(epochs_range, loss_val, label="Train Loss", linewidth=2)
# axes[1].plot(epochs_range, val_loss, label="Val Loss",   linewidth=2, linestyle="--")
# axes[1].axvline(x=15, color="gray", linestyle=":", label="Phase 2 start")
# axes[1].set_title("Loss")
# axes[1].set_xlabel("Epoch")
# axes[1].set_ylabel("Loss")
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)
#
# plt.suptitle("MobileNetV2 — Lung Cancer Classification", fontsize=14)
# plt.tight_layout()
# plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
# plt.show()
#
# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(7, 6))
# sns.heatmap(
#     cm, annot=True, fmt="d", cmap="Blues",
#     xticklabels=classes, yticklabels=classes,
#     linewidths=0.5
# )
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.tight_layout()
# plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
# plt.show()
#
# # Save final model
# model.save("lung_cancer_mobilenetv2_final.keras")
# print("\nModel saved: lung_cancer_mobilenetv2_final.keras")
# print("Best model also saved: best_lung_model.keras")

import os
import cv2
from collections import Counter
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
# location of the dataset
dataset_path = r"D:\lung_dataset\The IQ-OTHNCCD lung cancer dataset"
classes = ["Bengin cases", "Malignant cases", "Normal cases"]
#load data
data = []
labels = []
for category in classes:
    folder_path = os.path.join(dataset_path, category)
    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        image = cv2.imread(img_path)
        data.append(image)
        labels.append(category)
print("Total images:", len(data))
print("Total labels:", len(labels))

# plt.imshow(data[0])
# plt.title(labels[0])
# plt.show()
print(Counter(labels))

# check the image size
image_sizes = []
for img in data:
    image_sizes.append(img.shape)
unique_sizes = set(image_sizes)
print("Unique image sizes in dataset:")
print(unique_sizes)

# this is resize the size of image into all images fix size
# resized_data = []
# for img in data:
#     resized_img = cv2.resize(img, (224, 224))
#     resized_data.append(resized_img)
# X = np.array(resized_data)
# print("Dataset shape:", X.shape)

# resize + noise reduction
# processed_data = []
#
# for img in data:
#     resized_img = cv2.resize(img, (224, 224))
#
#     # Noise Reduction using Gaussian Blur
#     denoised_img = cv2.GaussianBlur(resized_img, (3,3), 0)
#
#     processed_data.append(denoised_img)
#
# X = np.array(processed_data)
#
# print("Dataset shape:", X.shape)
# resize + noise reduction + CLAHE
processed_data = []

# create CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for img in data:
    # Resize
    resized_img = cv2.resize(img, (224, 224))

    # Noise Reduction
    denoised_img = cv2.GaussianBlur(resized_img, (3, 3), 0)

    # Convert to grayscale for CLAHE
    gray = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE
    clahe_img = clahe.apply(gray)

    # Convert back to 3 channels
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

    processed_data.append(clahe_img)

X = np.array(processed_data)

print("Dataset shape:", X.shape)

import matplotlib.pyplot as plt

# Original image
original = cv2.resize(data[0], (224,224))

# Noise reduction
blur = cv2.GaussianBlur(original, (3,3), 0)

# CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Convert to grayscale
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# Apply CLAHE
clahe_img = clahe.apply(gray)

# Convert back to BGR
clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
plt.title("After Gaussian Blur")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
plt.title("After CLAHE")

plt.show()

# label encoding (0,1,2)
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
print("Encoded labels sample:", y[:10])
# show mapping
mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("Label Mapping:", mapping)
y = np.array(y)
print("Label shape:", y.shape)
# unique labels check
print("Unique labels:", np.unique(y))

# this is augmentation of the dataset folder bengin cases
augmented_images = []
augmented_labels = []

for img, label in zip(X, y):
    if label == mapping["Bengin cases"]:  # augment only benign class
        # horizontal flip
        flip = cv2.flip(img, 1)
        augmented_images.append(flip)
        augmented_labels.append(label)
        # rotate 90
        rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        augmented_images.append(rotate)
        augmented_labels.append(label)
        # brightness change
        bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
        augmented_images.append(bright)
        augmented_labels.append(label)
# add augmented data
X_aug = np.array(augmented_images)
y_aug = np.array(augmented_labels)
print("Augmented images:", X_aug.shape)
# combine original + augmented
X = np.concatenate((X, X_aug), axis=0)
y = np.concatenate((y, y_aug), axis=0)
print("Final dataset shape:", X.shape)
print("Final labels shape:", y.shape)

# normiization
X = X.astype("float32") / 255.0
print("After normalization")
print("Min pixel value:", X.min())
print("Max pixel value:", X.max())
# data shuffle
X, y = shuffle(X, y, random_state=42)
print("Dataset shuffled successfully")
print("X shape:", X.shape)
print("y shape:", y.shape)

# check that images is corrupted or not
corrupted_count = 0
for category in classes:
    folder_path = os.path.join(dataset_path, category)
    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        image = cv2.imread(img_path)
        if image is None:
            corrupted_count += 1
            print("Corrupted:", img_path)
print("Total corrupted images:", corrupted_count)