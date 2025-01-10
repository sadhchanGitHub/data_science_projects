import numpy as np
import os
import time
import logging
import sys

def verify_shape_of_splits(training_dir, category):
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
    
    # Log shapes
    logging.info(f"Train data shape for {category}: {train_data.shape}")
    logging.info(f"Train Labels shape for {category}: {train_labels.shape}")
    logging.info(f"Val data shape for {category}: {val_data.shape}")
    logging.info(f"Val Labels shape for {category}: {val_labels.shape}")
    logging.info(f"Test data shape for {category}: {test_data.shape}")
    logging.info(f"Test Labels shape for {category}: {test_labels.shape}")
    logging.info("\n")


#---
    
categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]

# Main execution block
# Main execution block
if __name__ == "__main__":
    timestamp = int(time.time())
    
    # Logging configuration
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"03_b_test_shape_of_npy_splits_{timestamp}.log")

    # Configure logging only in the main process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(" \n")
    logging.info("called via 03_b_test_shape_of_npy_splits.py...\n")
    logging.info(" Script Started ...\n")
    logging.info("This script will check the shape of .npy files for each category, split into train, val, and test...\n")

    # Check if image size is passed as an argument
    if len(sys.argv) != 2:
        print("Usage: python 03_b_test_shape_of_npy_splits.py <image_size>")
        sys.exit(1)

    # Get image size from the command-line argument
    image_size = int(sys.argv[1])
    logging.info(f"image_size is {image_size} \n")

    # Define paths
    preprocessed_dir = f"../data/preprocessed_data_{image_size}"
    training_dir = f"../data/training_data_{image_size}"

    # Process each category
    for category in categories:
        verify_shape_of_splits(training_dir, category)

    logging.info(" \n script done \n")

   
    