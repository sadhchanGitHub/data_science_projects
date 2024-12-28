import os
import numpy as np
import cv2
import logging
from dask import delayed, compute
from dask.distributed import Client
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
input_dir = "../data/training_data/train"  # Input directory
output_dir = "../data/training_data/train_resized_256"  # Output directory
os.makedirs(output_dir, exist_ok=True)

# Resize parameters
new_image_size = 256

# Categories
categories = [
    "AnnualCrop",
    "Forest",
    "Highway",
    "Industrial",
    "HerbaceousVegetation",
    "Residential"
]

# Function to process chunks of .npy files
@delayed
def process_chunk(input_path, output_path, start_idx, end_idx, resize_shape):
    try:
        # Load the .npy file
        data = np.load(input_path)
        chunk = data[start_idx:end_idx]

        # Resize the chunk
        resized_chunk = np.array([cv2.resize(img, (resize_shape, resize_shape), interpolation=cv2.INTER_AREA) for img in chunk])

        # Save resized chunk to output path
        np.save(output_path, resized_chunk)
        logging.info(f"Processed chunk {start_idx}:{end_idx} for {os.path.basename(input_path)}")
    except Exception as e:
        logging.error(f"Error processing chunk {start_idx}:{end_idx} of {input_path}: {e}")

# Function to split and process files
def process_npy_file(category, input_dir, output_dir, resize_shape, chunk_size=100):
    try:
        # Input and output paths
        img_file = os.path.join(input_dir, f"{category}_train.npy")
        mask_file = os.path.join(input_dir, f"{category}_train_masks_combined.npy")
        output_img_file = os.path.join(output_dir, f"{category}_train.npy")
        output_mask_file = os.path.join(output_dir, f"{category}_train_masks_combined.npy")

        # Load the .npy file to get the size
        images = np.load(img_file)
        masks = np.load(mask_file)
        num_samples = images.shape[0]

        # Create delayed tasks for chunks
        tasks = []
        for i in range(0, num_samples, chunk_size):
            img_chunk_path = output_img_file.replace(".npy", f"_chunk_{i}.npy")
            mask_chunk_path = output_mask_file.replace(".npy", f"_chunk_{i}.npy")
            tasks.append(process_chunk(img_file, img_chunk_path, i, min(i + chunk_size, num_samples), resize_shape))
            tasks.append(process_chunk(mask_file, mask_chunk_path, i, min(i + chunk_size, num_samples), resize_shape))

        # Execute the tasks
        compute(*tasks)

        # Merge chunks into a single .npy file
        merge_chunks(output_img_file)
        merge_chunks(output_mask_file)

        logging.info(f"Finished processing {category}")
    except Exception as e:
        logging.error(f"Error processing category {category}: {e}")

# Function to merge chunks into a single file
def merge_chunks(output_path):
    try:
        chunk_files = sorted([f for f in os.listdir(os.path.dirname(output_path)) if f.startswith(os.path.basename(output_path).replace(".npy", "_chunk"))])
        merged_data = np.concatenate([np.load(os.path.join(os.path.dirname(output_path), chunk_file)) for chunk_file in chunk_files], axis=0)
        np.save(output_path, merged_data)

        # Delete chunk files after merging
        for chunk_file in chunk_files:
            os.remove(os.path.join(os.path.dirname(output_path), chunk_file))
        logging.info(f"Merged and saved final file to {output_path}")
    except Exception as e:
        logging.error(f"Error merging chunks for {output_path}: {e}")

# Main function
def main():
    # Initialize Dask client
    client = Client()
    logging.info(f"Dask client started: {client}")

    for category in categories:
        process_npy_file(category, input_dir, output_dir, new_image_size)

    # Close Dask client
    client.close()
    logging.info("Dask client closed. Resizing complete.")

if __name__ == "__main__":
    main()
