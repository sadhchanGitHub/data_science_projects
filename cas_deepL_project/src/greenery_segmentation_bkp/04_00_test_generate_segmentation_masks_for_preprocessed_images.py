import os
import kagglehub
import tensorflow as tf
import numpy as np

# Define paths
model_save_dir = "../models/deeplabv3"
model_name = "1.tflite"
os.makedirs(model_save_dir, exist_ok=True)
saved_model_path = os.path.join(model_save_dir, model_name)


# Define paths
preprocessed_outdir = "../data/preprocessed_output_data"
output_masks_dir = "../data/segmentation_masks_test"
os.makedirs(output_masks_dir, exist_ok=True)
"""
import kagglehub

# Download latest version
path = kagglehub.model_download("tensorflow/deeplabv3/tfLite/default")
saved_model_path = "../models/deeplabv3/model.tflite"

print("Path to model files:", path)

model is downloaded and saves, as it is saving in some cache folders
"""
    

# Load the TensorFlow Lite model
# Load the TensorFlow Lite model
def load_tflite_model():
    if not os.path.exists(saved_model_path):
        raise FileNotFoundError(f"Model not found at {saved_model_path}. Ensure it is downloaded and saved.")
    
    interpreter = tf.lite.Interpreter(model_path=saved_model_path)
    interpreter.allocate_tensors()
    print("TensorFlow Lite model loaded successfully!")
    return interpreter

# Main process
def process_images():

    categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']  # Get expected input shape dynamically

    for category in categories:
        category_dir = os.path.join(preprocessed_outdir, f"{category}_train.npy")
        output_category_dir = os.path.join(output_masks_dir, category)
        os.makedirs(output_category_dir, exist_ok=True)

        print(f"Processing category: {category}")
        images = np.load(category_dir)

        for i, img in enumerate(images[:10]):  # Process only the first 10 images
            img = preprocess_image(img, input_shape)
            
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            mask = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
            
            output_path = os.path.join(output_category_dir, f"mask_{i}.npy")
            save_mask(mask, output_path)

# Example preprocessing and saving functions
def preprocess_image(image, input_shape):
    target_size = (input_shape[1], input_shape[2])  # Extract height and width from input shape
    img = tf.image.resize(image, target_size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return np.expand_dims(img, axis=0)  # Add batch dimension

def save_mask(mask, output_path):
    mask = mask.astype(np.uint8)
    np.save(output_path, mask)
    print(f"Saved segmentation mask to {output_path}")

if __name__ == "__main__":
    process_images()
