import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Define paths
preprocessed_dir = "../data/preprocessed_data_imSize513"
output_masks_dir = "../data/segmentation_masks"
os.makedirs(output_masks_dir, exist_ok=True)

# Load the pre-trained DeepLabV3+ model
base_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet')
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Conv2D(21, (1, 1), activation='softmax')  # Example: 21 classes for COCO dataset
])
print("Model loaded successfully!")

# Function to preprocess an image for the model
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(513, 513))
    img = img_to_array(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = imagenet_utils.preprocess_input(img)
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to save the segmentation mask
def save_mask(mask, output_path):
    mask = np.argmax(mask, axis=-1).astype(np.uint8)  # Convert probabilities to class indices
    np.save(output_path, mask)
    print(f"Saved segmentation mask to {output_path}")

# Process each category
categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]
for category in categories:
    category_dir = os.path.join(preprocessed_dir, f"{category}_train.npy")  # Adjust paths as needed
    output_category_dir = os.path.join(output_masks_dir, category)
    os.makedirs(output_category_dir, exist_ok=True)

    # Load preprocessed images
    print(f"Processing category: {category}")
    images = np.load(category_dir)

    for i, img in enumerate(images):
        img_batch = np.expand_dims(img, axis=0)  # Add batch dimension
        predictions = model.predict(img_batch)
        output_path = os.path.join(output_category_dir, f"mask_{i}.npy")
        save_mask(predictions[0], output_path)

print("Segmentation masks generated successfully!")
