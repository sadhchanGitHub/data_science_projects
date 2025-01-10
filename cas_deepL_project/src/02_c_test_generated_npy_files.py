# save as single .npy file
# check if the saved .npy file has all the images and its quality etc..


import numpy as np
import os
import time
import logging
import sys


# Categories
categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]

# Main execution block
if __name__ == "__main__":
    timestamp = int(time.time())
    
    # Logging configuration
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"02_c_test_generated_npy_files_{timestamp}.log")

    # Configure logging only in the main process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(" \n")
    logging.info("called via 02_c_test_generated_npy_files.py...\n")
    logging.info(" Script Started ...\n")
    logging.info("This script will check the shape of npy file per category ...\n")

    # Check if image size is passed as an argument
    if len(sys.argv) != 2:
        print("Usage: python 02_b_combine_multiple_npy_into_one.py <image_size>")
        sys.exit(1)

    # Get image size from the command-line argument
    image_size = int(sys.argv[1])
    logging.info(f"image_size is {image_size}")

    # Define paths
    preprocessed_dir = f"../data/preprocessed_data_{image_size}"

    # Combine files for each category
    for category in categories:
        combined_file_path = os.path.join(preprocessed_dir, f"{category}.npy")
        data = np.load(combined_file_path, allow_pickle=True)
        logging.info(f"Shape of combined data for {category}: {data.shape}")

    logging.info("script done \n")
    

