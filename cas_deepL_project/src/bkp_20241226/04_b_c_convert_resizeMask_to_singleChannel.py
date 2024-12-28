import os
import numpy as np

# Paths
multi_channel_masks_dir = "../data/resized_masks_513"
single_channel_masks_dir = "../data/single_channel_masks"
os.makedirs(single_channel_masks_dir, exist_ok=True)

categories = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Residential"]
splits = ["train", "val", "test"]

# Conversion function
def convert_to_single_channel(multi_channel_batch):
    return np.argmax(multi_channel_batch, axis=-1).astype(np.uint8)

# Process masks
for category in categories:
    for split in splits:
        input_dir = os.path.join(multi_channel_masks_dir, category, split)
        output_dir = os.path.join(single_channel_masks_dir, category, split)
        os.makedirs(output_dir, exist_ok=True)

        batch_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
        for batch_file in batch_files:
            multi_channel_batch = np.load(os.path.join(input_dir, batch_file))
            single_channel_batch = np.array([convert_to_single_channel(mask) for mask in multi_channel_batch])
            output_path = os.path.join(output_dir, batch_file)
            np.save(output_path, single_channel_batch)
            print(f"Converted and saved: {output_path}")
