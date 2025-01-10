import os
import time
import logging
import sys
import numpy as np
import cv2
from dask import delayed
from dask.distributed import Client
import multiprocessing

# Fix multiprocessing start method (Unix only)
multiprocessing.set_start_method("fork", force=True)

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
def process_and_save_batch(batch_files, batch_idx, category):
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
        batch_files = files[i:i + batch_size]
        tasks.append(process_and_save_batch(batch_files, i // batch_size, category))

    # Execute Dask tasks
    from dask import compute
    compute(*tasks)
    logging.info(f"Finished processing category: {category}")

# Categories
categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]

# Main execution block
if __name__ == "__main__":
    timestamp = int(time.time())
    
    # Logging configuration
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"02_a_load_preprocess_dask_images_resizes_{timestamp}.log")

    # Configure logging only in the main process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(" \n")
    logging.info("called via 02_a_load_preprocess_dask_images_resize.py...\n")
    logging.info(" Script Started ...\n")
    logging.info("This script will resize the input eurosat images which are 64x64 to the image_size passed as input argument, it will use DASK to avoid memory issues...\n")

    # Check if image size is passed as an argument
    if len(sys.argv) != 2:
        print("Usage: python 02_a_load_preprocess_dask_images_resize.py <image_size>")
        sys.exit(1)

    # Get image size from the command-line argument
    image_size = int(sys.argv[1])
    logging.info(f"image_size is {image_size}")

    # Paths
    dataset_path = "../data/selected_categories/EuroSAT"
    preprocessed_dir = f"../data/preprocessed_data_{image_size}"
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Initialize Dask client with custom dashboard port
    client = Client(dashboard_address=":8788")
    logging.info(f"Dask client started: {client}")

    # Process images
    for category in categories:
        try:
            process_category(category)
        except Exception as e:
            logging.error(f"Error processing category {category}: {e}")

    # Close Dask client
    client.close()
    logging.info("Dask client closed.")
    logging.info("script done \n")
