import os
import cv2
import numpy as np
import logging
from dask import delayed
from dask.distributed import Client
import multiprocessing

# Fix multiprocessing start method (Unix only)
multiprocessing.set_start_method("fork", force=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
dataset_path = "../data/selected_categories/EuroSAT"
preprocessed_dir = "../data/preprocessed_data_imSize256"
os.makedirs(preprocessed_dir, exist_ok=True)


image_size = 256
#category = "AnnualCrop"
categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]

# Image processing function
def process_image(file_path):
    try:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (image_size, image_size))
        return img / 255.0
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None

# Batch processing function with Dask
@delayed
def process_and_save_batch(batch_files, batch_idx):
    batch_images = [process_image(file_path) for file_path in batch_files]
    batch_images = [img for img in batch_images if img is not None]
    batch_output_path = os.path.join(preprocessed_dir, f"{category}_batch_{batch_idx}.npy")
    np.save(batch_output_path, np.array(batch_images))
    logging.info(f"Saved batch {batch_idx} to {batch_output_path}")

def process_category(category):
    category_path = os.path.join(dataset_path, category)
    files = [os.path.join(category_path, file) for file in os.listdir(category_path)]
    batch_size = 20
    logging.info(f"Processing category: {category}")
    tasks = []

    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        tasks.append(process_and_save_batch(batch_files, i // batch_size))

    # Execute Dask tasks
    from dask import compute
    compute(*tasks)
    logging.info(f"Finished processing category: {category}")

# Main execution block
if __name__ == '__main__':
    # Initialize Dask client with custom dashboard port
    client = Client(dashboard_address=":8788")
    logging.info(f"Dask client started: {client}")

    # Process images
    for category in categories:
        process_category(category)

    # Close Dask client
    client.close()
    logging.info("Dask client closed.")
