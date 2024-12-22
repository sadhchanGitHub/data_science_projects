
import os
import cv2
import numpy as np
import logging
from multiprocessing import Pool
import gc  # For memory cleanup

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
dataset_path = "../data/selected_categories/EuroSAT"
preprocessed_dir = "../data/preprocessed_data_imSize513"
os.makedirs(preprocessed_dir, exist_ok=True)

# Image size
image_size = 513  # Change this if needed

# Single category to process
category = "Forest"  # Replace with the actual category name

# Function to process a single image
def process_image(file_path):
    try:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (image_size, image_size))
        return img / 255.0  # Normalize to [0, 1]
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None

# Function to log memory usage
def log_memory_usage():
    import psutil
    memory = psutil.virtual_memory()
    logging.info(f"Memory Usage: {memory.percent}% used, {memory.available // (1024**2)} MB available")

# Function to process all images in the category
def process_category(category):
    category_path = os.path.join(dataset_path, category)
    files = [os.path.join(category_path, file) for file in os.listdir(category_path)]
    batch_size = 20  # Smaller batch size to reduce memory pressure

    logging.info(f"Processing category: {category}")
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        with Pool(processes=2) as pool:  # Adjust based on system resources
            batch_images = list(pool.map(process_image, batch_files))
            batch_images = [img for img in batch_images if img is not None]  # Filter out failed images
        
        # Save batch to temporary file
        batch_output_path = os.path.join(preprocessed_dir, f"{category}_batch_{i // batch_size}.npy")
        np.save(batch_output_path, np.array(batch_images))
        logging.info(f"Saved batch {i // batch_size + 1} to {batch_output_path}")
        
        # Clear memory and log status
        log_memory_usage()
        gc.collect()

# Function to combine all batch files into one
def combine_batches(category):
    batch_files = [os.path.join(preprocessed_dir, f) for f in os.listdir(preprocessed_dir) if f.startswith(category) and "batch" in f]
    all_images = []
    for batch_file in sorted(batch_files):  # Ensure proper order
        all_images.extend(np.load(batch_file))
    final_output_path = os.path.join(preprocessed_dir, f"{category}.npy")
    np.save(final_output_path, np.array(all_images))
    logging.info(f"Combined all batches into {final_output_path}")

    # Optionally, delete temporary batch files
    for batch_file in batch_files:
        os.remove(batch_file)
        logging.info(f"Deleted temporary file: {batch_file}")

# Main script
if __name__ == "__main__":
    process_category(category)
    combine_batches(category)
