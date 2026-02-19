import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# a. Load the MNIST handwritten digit dataset
# ---------------------------------------------------------
# Loading the standard MNIST dataset (identical to the repo provided)
# Default split: 60,000 Training images, 10,000 Test images
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

print(f"Original Train Shape: {x_train_full.shape}, Test Shape: {x_test.shape}")

# ---------------------------------------------------------
# b. Normalize the pixel values of the images
# ---------------------------------------------------------
# Convert integers to floats and scale from [0, 255] to [0, 1]
# This helps gradient descent converge faster during training
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# ---------------------------------------------------------
# c. Apply one-hot encoding to the target labels
# ---------------------------------------------------------
# Converts class vectors (integers) to binary class matrices
# Example: 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
y_train_full = to_categorical(y_train_full, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# ---------------------------------------------------------
# d. Split the data into training, validation, and test sets
# ---------------------------------------------------------
# The original dataset only has Train and Test. 
# We need to carve out a Validation set from the original Training data.
# We will take 10,000 images (approx 16.67%) for validation.
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, 
    y_train_full, 
    test_size=0.1667,  # 10,000 / 60,000
    random_state=42,   # Ensures reproducibility
    stratify=y_train_full # Keeps class distribution balanced
)

# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------
print("\n--- Final Dataset Shapes ---")
print(f"Training Set:   Images: {x_train.shape}, Labels: {y_train.shape}")
print(f"Validation Set: Images: {x_val.shape},   Labels: {y_val.shape}")
print(f"Test Set:       Images: {x_test.shape},  Labels: {y_test.shape}")

# Optional: Reshape for CNN (28, 28, 1) or Dense (784,)
# x_train = x_train.reshape(-1, 28, 28, 1) 