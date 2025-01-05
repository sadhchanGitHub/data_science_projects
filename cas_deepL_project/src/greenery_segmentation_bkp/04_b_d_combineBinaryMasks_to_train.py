import os
import numpy as np
import logging

# Paths
resized_masks_dir = "../data/resized_binary_masks_256"  # Path to resized masks
combined_masks_dir = "../data/training_data"  # Path to save combined masks
os.makedirs(combined_masks_dir, exist_ok=True)

categories = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Residential"]
splits = ["train", "val", "test"]

# Function to combine resized masks into a single file
def combine_resized_masks(category, split):
    input_dir = os.path.join(resized_masks_dir, category, split)
    mask_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy')])

    combined_masks = []
    for file_path in mask_files:
        batch_masks = np.load(file_path)
        combined_masks.append(batch_masks)

    combined_array = np.concatenate(combined_masks, axis=0)  # Combine along batch dimension
    logging.info(f"Combined masks shape for {category} ({split}): {combined_array.shape}")

    output_path = os.path.join(combined_masks_dir, f"{category}_{split}_masks_combined.npy")
    np.save(output_path, combined_array)
    logging.info(f"Saved combined masks for {category} ({split}) at {output_path}")

# Main execution block
if __name__ == '__main__':
    for category in categories:
        for split in splits:
            logging.info(f"Processing {category} ({split})...")
            combine_resized_masks(category, split)
