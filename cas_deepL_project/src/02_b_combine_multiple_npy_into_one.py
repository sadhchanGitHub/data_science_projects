import numpy as np
import os

# Define paths
preprocessed_dir = "../data/preprocessed_data_imSize513"
preprocessed_output_dir = "../data/preprocessed_output_data"
os.makedirs(preprocessed_output_dir, exist_ok=True)

def combine_npy_files(category):
    # Get all batch files for the category
    batch_files = sorted(
        [os.path.join(preprocessed_dir, file) 
         for file in os.listdir(preprocessed_dir) 
         if file.startswith(f"{category}_batch_") and file.endswith(".npy")]
    )
    
    # Determine the total number of images
    total_images = 0
    sample_shape = None
    for batch_file in batch_files:
        batch_data = np.load(batch_file, allow_pickle=True)
        total_images += len(batch_data)
        if sample_shape is None:
            sample_shape = batch_data[0].shape  # Get shape of one sample
    
    print(f"Total images for {category}: {total_images}")
    print(f"Sample image shape: {sample_shape}")
    
    # Create a memmapped file
    combined_file_path = os.path.join(preprocessed_output_dir, f"{category}.npy")
    combined_array = np.lib.format.open_memmap(
        combined_file_path,
        mode='w+',
        dtype='float32',
        shape=(total_images, *sample_shape)
    )
    
    # Append batches to the memmapped file
    start_idx = 0
    for batch_file in batch_files:
        print(f"Loading {batch_file}")
        batch_data = np.load(batch_file, allow_pickle=True)
        batch_size = len(batch_data)
        combined_array[start_idx:start_idx+batch_size] = batch_data
        start_idx += batch_size
        print(f"Appended {batch_file} to {combined_file_path}")
    
    print(f"Finished combining files for {category} into {combined_file_path}")

# Categories
categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]

# Combine files for each category
for category in categories:
    combine_npy_files(category)
