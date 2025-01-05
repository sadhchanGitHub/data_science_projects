import os
import numpy as np
import cv2
import logging
from dask import delayed, compute
from dask.distributed import Client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
binary_masks_dir = "../data/binary_masks"  # Input directory with binary masks
resized_masks_dir = "../data/resized_binary_masks_256"  # Output directory for resized masks
os.makedirs(resized_masks_dir, exist_ok=True)

image_size = 256
categories = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Residential"]
splits = ["train", "val", "test"]
batch_size = 20

# Mask resizing function
def resize_mask(file_path):
    try:
        mask = np.load(file_path)  # Load the .npy mask
        resized_mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        return resized_mask
    except Exception as e:
        logging.error(f"Error resizing mask {file_path}: {e}")
        return None

# Dask-based batch processing function
@delayed
def resize_and_save_batch(batch_files, batch_idx, output_dir):
    resized_masks = []
    for file_path in batch_files:
        resized_mask = resize_mask(file_path)
        if resized_mask is not None:
            resized_masks.append(resized_mask)
        else:
            logging.error(f"Skipped file: {file_path}")
    if resized_masks:
        batch_output_path = os.path.join(output_dir, f"batch_{batch_idx}.npy")
        os.makedirs(os.path.dirname(batch_output_path), exist_ok=True)
        np.save(batch_output_path, np.array(resized_masks, dtype=np.uint8))
        logging.info(f"Saved batch {batch_idx} to {batch_output_path}")

# Function to resize masks for a single category and split
def resize_split(category, split):
    input_dir = os.path.join(binary_masks_dir, category, split)
    output_dir = os.path.join(resized_masks_dir, category, split)
    os.makedirs(output_dir, exist_ok=True)

    mask_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.npy')]
    logging.info(f"Resizing {split} split for category: {category} with {len(mask_files)} files.")

    tasks = []
    for i in range(0, len(mask_files), batch_size):
        batch_files = mask_files[i:i + batch_size]
        tasks.append(resize_and_save_batch(batch_files, i // batch_size, output_dir))

    # Execute Dask tasks
    compute(*tasks)
    logging.info(f"Finished resizing {split} split for category: {category}")

# Main execution block
if __name__ == '__main__':
    # Initialize Dask client
    client = Client()
    logging.info(f"Dask client started: {client}")

    # Resize masks for all categories and splits
    for category in categories:
        for split in splits:
            resize_split(category, split)

    # Close Dask client
    client.close()
    logging.info("Dask client closed.")
