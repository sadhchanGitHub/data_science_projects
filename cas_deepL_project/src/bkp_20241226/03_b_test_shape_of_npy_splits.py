import numpy as np
import os

# Verify combined data
preprocessed_output_dir = "../data/preprocessed_output_data"
categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]

for category in categories:
    train_data_path = f"{preprocessed_output_dir}/{category}_train.npy"
    val_data_path = f"{preprocessed_output_dir}/{category}_val.npy"
    test_data_path = f"{preprocessed_output_dir}/{category}_test.npy"

    # Load using NumPy
    train_data = np.load(train_data_path)
    val_data = np.load(val_data_path)
    test_data = np.load(test_data_path)

    print(f"Train data shape for {category}: {train_data.shape}")
    print(f"Val data shape for {category}: {val_data.shape}")
    print(f"Test data shape for {category}: {test_data.shape}")
