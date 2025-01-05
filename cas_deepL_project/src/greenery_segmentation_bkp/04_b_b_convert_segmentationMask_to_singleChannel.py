import os
import numpy as np
import logging
from multiprocessing import Pool
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Paths
multi_channel_masks_dir = "../data/segmentation_masks"  # Input multi-channel masks
single_channel_masks_dir = "../data/single_channel_masks"  # Output single-channel masks
binary_masks_dir = "../data/binary_masks"  # Output binary masks

# Categories and splits
categories = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Residential"]
splits = ["train", "val", "test"]

# Ensure output directories exist
for category in categories:
    for split in splits:
        os.makedirs(os.path.join(single_channel_masks_dir, category, split), exist_ok=True)
        os.makedirs(os.path.join(binary_masks_dir, category, split), exist_ok=True)

def convert_to_binary(single_channel_mask, foreground_class=1):
    """
    Converts a single-channel mask to a binary mask.
    Foreground pixels are set to 1, and background pixels are set to 0.
    """
    return (single_channel_mask == foreground_class).astype(np.uint8)

def process_single_channel_batch(args):
    """
    Process a multi-channel batch file: convert to single-channel masks and save.
    """
    batch_file, input_dir, output_dir = args
    try:
        # Load multi-channel batch
        batch_path = os.path.join(input_dir, batch_file)
        multi_channel_batch = np.load(batch_path)
        logging.info(f"Processing batch: {batch_file}, shape: {multi_channel_batch.shape}")

        # Convert to single-channel masks
        single_channel_batch = np.array([np.argmax(mask, axis=-1).astype(np.uint8) for mask in multi_channel_batch])

        # Save single-channel masks
        output_path = os.path.join(output_dir, batch_file)
        np.save(output_path, single_channel_batch)
        logging.info(f"Processed and saved single-channel masks: {output_path}")

    except Exception as e:
        logging.error(f"Error processing {batch_file}: {e}")

def process_binary_batch(args):
    """
    Process a single-channel batch file: convert to binary masks and save.
    """
    batch_file, input_dir, output_dir, foreground_class = args
    try:
        # Load single-channel batch
        batch_path = os.path.join(input_dir, batch_file)
        single_channel_batch = np.load(batch_path)

        # Convert to binary masks
        binary_batch = np.array([convert_to_binary(mask, foreground_class) for mask in single_channel_batch])

        # Skip saving if the entire batch is empty
        if np.max(binary_batch) == 0:
            logging.info(f"Skipping batch {batch_file} with no useful binary masks.")
            return

        # Save binary masks
        output_path = os.path.join(output_dir, batch_file)
        np.save(output_path, binary_batch)
        logging.info(f"Processed and saved binary masks: {output_path}")

    except Exception as e:
        logging.error(f"Error processing {batch_file}: {e}")

# Main script
if __name__ == "__main__":
    start_time = time.time()
    
    # Step 1: Convert multi-channel masks to single-channel masks
    for category in categories:
        for split in splits:
            input_dir = os.path.join(multi_channel_masks_dir, category, split)
            output_dir = os.path.join(single_channel_masks_dir, category, split)

            # List all .npy files in the input directory
            batch_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

            # Prepare arguments for parallel processing
            args = [(batch_file, input_dir, output_dir) for batch_file in batch_files]

            # Process files in parallel
            with Pool(processes=4) as pool:
                pool.map(process_single_channel_batch, args)

    # Step 2: Convert single-channel masks to binary masks
    for category in categories:
        for split in splits:
            input_dir = os.path.join(single_channel_masks_dir, category, split)
            output_dir = os.path.join(binary_masks_dir, category, split)

            # List all .npy files in the input directory
            batch_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

            # Prepare arguments for parallel processing
            args = [(batch_file, input_dir, output_dir, 1) for batch_file in batch_files]  # Foreground class = 1

            # Process files in parallel
            with Pool(processes=4) as pool:
                pool.map(process_binary_batch, args)

    logging.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
