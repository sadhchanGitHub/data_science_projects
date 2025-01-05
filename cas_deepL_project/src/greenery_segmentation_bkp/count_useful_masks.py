import os
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Paths
multi_channel_masks_dir = "../data/segmentation_masks"  # Input multi-channel masks
single_channel_masks_dir = "../data/single_channel_masks"  # Output single-channel masks

# Categories and splits
categories = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Residential"]
splits = ["train", "val", "test"]

# Initialize counters
useful_mask_count = 0
total_mask_count = 0

# Count useful masks
for category in categories:
    for split in splits:
        output_dir = os.path.join(single_channel_masks_dir, category, split)
        for batch_file in os.listdir(output_dir):
            single_channel_batch = np.load(os.path.join(output_dir, batch_file))
            non_empty_masks = [mask for mask in single_channel_batch if np.max(mask) > 0]
            useful_mask_count += len(non_empty_masks)
            total_mask_count += len(single_channel_batch)
            logging.info(f"{batch_file}: {len(non_empty_masks)} useful masks out of {len(single_channel_batch)}")

# Summary of results
logging.info(f"Total useful masks: {useful_mask_count} / {total_mask_count}")
