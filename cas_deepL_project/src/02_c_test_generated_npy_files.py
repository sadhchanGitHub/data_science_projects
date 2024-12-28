# save as single .npy file
# check if the saved .npy file has all the images and its quality etc..

import numpy as np
import os

# Load and verify
preprocessed_dir = "../data/preprocessed_data_imSize256"
# Categories
categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]


# Combine files for each category
for category in categories:
    combined_file_path = os.path.join(preprocessed_dir, f"{category}.npy")
    data = np.load(combined_file_path, allow_pickle=True)
    print(f"Shape of combined data for {category}: {data.shape}")

