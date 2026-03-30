import os
import cv2
from collections import Counter
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder


# location of the dataset
dataset_path = "lung_dataset\\The IQ-OTHNCCD lung cancer dataset"
classes = ["Bengin cases", "Malignant cases", "Normal cases"]
#load data
data = []
labels = []
for category in classes:
    folder_path = os.path.join(dataset_path, category)
    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        image = cv2.imread(img_path)
        if image is None:
            print("Skipped:", img_path)
            continue
        data.append(image)
        labels.append(category)
print("Total images:", len(data))
print("Total labels:", len(labels))

print(Counter(labels))

# check the image size
image_sizes = []
for img in data:
    image_sizes.append(img.shape)
unique_sizes = set(image_sizes)
print("Unique image sizes in dataset:")
print(unique_sizes)

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

# Original image
original = cv2.resize(data[0], (224,224))
# Noise reduction
blur = cv2.GaussianBlur(original, (3,3), 0)
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
