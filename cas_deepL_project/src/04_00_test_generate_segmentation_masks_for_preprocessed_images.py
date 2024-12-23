import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# Define paths
preprocessed_dir = "../data/preprocessed_data_imSize513"
output_masks_dir = "../data/segmentation_masks_test"
os.makedirs(output_masks_dir, exist_ok=True)

# Load the pre-trained DeepLabV3+ model
model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet')  # Replace DenseNet121 if needed
print("Model loaded successfully!")

# Function to preprocess an image for the model
def preprocess_image(image_array):
    img = tf.image.resize(image_array, (513, 513))
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to save the segmentation mask
def save_mask(mask, output_path):
    mask = np.argmax(mask, axis=-1).astype(np.uint8)  # Convert probabilities to class indices
    np.save(output_path, mask)
    print(f"Saved segmentation mask to {output_path}")

# Select a category to test
category = "AnnualCrop"  # Change as needed
category_dir = os.path.join(preprocessed_dir, f"{category}_train.npy")
output_category_dir = os.path.join(output_masks_dir, category)
os.makedirs(output_category_dir, exist_ok=True)

# Load preprocessed images
print(f"Testing on category: {category}")
images = np.load(category_dir)

# Test on a few random images
num_test_images = 10
test_images = images[:num_test_images]

for i, img in enumerate(test_images):
    img_batch = preprocess_image(img)  # Preprocess the image
    predictions = model.predict(img_batch)  # Predict segmentation
    output_path = os.path.join(output_category_dir, f"mask_test_{i}.npy")
    save_mask(predictions[0], output_path)

print("Testing completed for a small batch of images!")
