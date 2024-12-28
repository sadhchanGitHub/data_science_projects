import os
import numpy as np
import cv2
import logging
from dask import delayed
from dask.distributed import Client
import multiprocessing

# Fix multiprocessing start method (Unix only)
multiprocessing.set_start_method("fork", force=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
masks_dir = "../data/segmentation_masks"  # Input directory with masks
resized_masks_dir = "../data/resized_masks_513"  # Output directory for resized masks
os.makedirs(resized_masks_dir, exist_ok=True)

image_size = 513
categories = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Residential"]
splits = ["train", "val", "test"]

# Mask processing function
def process_mask(file_path):
    try:
        mask = np.load(file_path)  # Load the .npy mask
        mask_resized = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        return mask_resized
    except Exception as e:
        logging.error(f"Error processing mask {file_path}: {e}")
        return None

# Batch processing function with Dask
@delayed
def process_and_save_batch(batch_files, batch_idx, output_dir):
    batch_masks = [process_mask(file_path) for file_path in batch_files]
    batch_masks = [mask for mask in batch_masks if mask is not None]
    batch_output_path = os.path.join(output_dir, f"batch_{batch_idx}.npy")
    os.makedirs(os.path.dirname(batch_output_path), exist_ok=True)
    np.save(batch_output_path, np.array(batch_masks, dtype=np.uint8))
    logging.info(f"Saved batch {batch_idx} to {batch_output_path}")

# Function to process masks for a single category and split
def process_split(category, split):
    input_dir = os.path.join(masks_dir, category, split)
    output_dir = os.path.join(resized_masks_dir, category, split)
    os.makedirs(output_dir, exist_ok=True)

    mask_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.npy')]
    batch_size = 20
    logging.info(f"Processing {split} split for category: {category}")
    tasks = []

    for i in range(0, len(mask_files), batch_size):
        batch_files = mask_files[i:i+batch_size]
        tasks.append(process_and_save_batch(batch_files, i // batch_size, output_dir))

    # Execute Dask tasks
    from dask import compute
    compute(*tasks)
    logging.info(f"Finished processing {split} split for category: {category}")

# Main execution block
if __name__ == '__main__':
    # Initialize Dask client with custom dashboard port
    client = Client(dashboard_address=":8788")
    logging.info(f"Dask client started: {client}")

    # Process each category and split
    for category in categories:
        for split in splits:
            process_split(category, split)

    # Close Dask client
    client.close()
    logging.info("Dask client closed.")
