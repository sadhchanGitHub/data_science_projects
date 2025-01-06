import numpy as np
import os

# Verify combined data
training_dir =  "../data/training_data"
categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]

for category in categories:
    train_data_path = f"{training_dir}/{category}_train.npy"
    train_labels_path = f"{training_dir}/{category}_train_labels.npy"
    val_data_path = f"{training_dir}/{category}_val.npy"
    val_labels_path = f"{training_dir}/{category}_val_labels.npy"
    test_data_path = f"{training_dir}/{category}_test.npy"
    test_labels_path = f"{training_dir}/{category}_test_labels.npy"

    # Load using NumPy
    train_data = np.load(train_data_path)
    train_labels = np.load(train_labels_path)
    val_data = np.load(val_data_path)
    val_labels = np.load(val_labels_path)
    test_data = np.load(test_data_path)
    test_labels = np.load(test_labels_path)


    print(f"Train data shape for {category}: {train_data.shape}")
    print(f"Train Labels shape for {category}: {train_labels.shape}")
    print(f"Val data shape for {category}: {val_data.shape}")
    print(f"Val Labels shape for {category}: {val_labels.shape}")
    print(f"Test data shape for {category}: {test_data.shape}")
    print(f"Test Labels shape for {category}: {test_labels.shape}")
    print("\n")