import os
import time
import logging
import sys

import numpy as np
import dask.array as da

# Function to split data incrementally
def split_data_dask(X, y, test_size, val_size):
    total_samples = X.shape[0]
    test_samples = int(total_samples * test_size)
    val_samples = int((total_samples - test_samples) * val_size)

    indices = np.random.permutation(total_samples)
    test_indices = indices[:test_samples]
    val_indices = indices[test_samples:test_samples + val_samples]
    train_indices = indices[test_samples + val_samples:]

    return (
        X[test_indices], y[test_indices],
        X[val_indices], y[val_indices],
        X[train_indices], y[train_indices]
    )


# Process and save splits incrementally
def process_and_save_category(preprocessed_dir, training_dir, category, test_size=0.2, val_size=0.2):
    file_path = os.path.join(preprocessed_dir, f"{category}.npy")
    logging.info(f"Processing {category}...")
    
    # Load combined `.npy` file as Dask array
    images = da.from_array(np.load(file_path, mmap_mode='r'), chunks="auto")
    labels = da.full((images.shape[0],), categories.index(category), chunks="auto")
    
    # Split data
    X_test, y_test, X_val, y_val, X_train, y_train = split_data_dask(images, labels, test_size, val_size)

    # Ensure the training directory exists
    os.makedirs(training_dir, exist_ok=True)

    # Save each split as a single `.npy` file
    for split_name, split_data, split_labels in [
        ("train", X_train, y_train), 
        ("val", X_val, y_val), 
        ("test", X_test, y_test)
    ]:
        output_data_path = os.path.join(training_dir, f"{category}_{split_name}.npy")
        output_label_path = os.path.join(training_dir, f"{category}_{split_name}_labels.npy")
        
        # Convert Dask arrays to NumPy arrays and save
        np.save(output_data_path, split_data.compute())
        np.save(output_label_path, split_labels.compute())

        logging.info(f"Saved {split_name} split for {category} to {output_data_path}")


categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]

# Main execution block
if __name__ == "__main__":
    timestamp = int(time.time())
    
    # Logging configuration
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"03_a_split_and_save_TrainValTest_per_category_for_preprocessed_images_{timestamp}.log")

    # Configure logging only in the main process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(" \n")
    logging.info("called via 03_a_split_and_save_TrainValTest_per_category_for_preprocessed_images_.py...\n")
    logging.info(" Script Started ...\n")
    logging.info("This script will split the single .npy file per category into category_train, category_val and category_test ...\n")

    # Check if image size is passed as an argument
    if len(sys.argv) != 2:
        print("Usage: python 03_a_split_and_save_TrainValTest_per_category_for_preprocessed_images_.py <image_size>")
        sys.exit(1)

    # Get image size from the command-line argument
    image_size = int(sys.argv[1])
    logging.info(f"image_size is {image_size} \n")

    # Define paths
    preprocessed_dir = f"../data/preprocessed_data_{image_size}"
    training_dir =  f"../data/training_data_{image_size}"
    os.makedirs(log_dir, exist_ok=True)

    # Process each category
    for category in categories:
        process_and_save_category(preprocessed_dir, training_dir, category)

    logging.info(" \n script done \n")
    

