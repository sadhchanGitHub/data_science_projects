import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import time
import os
from datetime import datetime
import sys
import shutil


timestamp = int(time.time())

def remap_labels_and_copy_images(training_dir, training_dir_remapped, category, split):

    logging.info(f"started remap_labels_and_copyimages for {category} {split} \n")
    
    os.makedirs(training_dir_remapped, exist_ok=True)

    # Load labels
    category_labels = np.load(f"{training_dir}/{category}_{split}_labels.npy")
    remapped_labels = np.ones_like(category_labels) if category == "HerbaceousVegetation" else np.zeros_like(category_labels)
    logging.info(f"remapping of the labels done for {category} {split} \n")
    
    np.save(f"{training_dir_remapped}/{category}_{split}_labels.npy", remapped_labels)
    logging.info(f"Saved remapped labels to {training_dir_remapped}/{category}_{split}_labels.npy")


    images_src = os.path.join(training_dir, f"{category}_{split}.npy")
    images_dst = os.path.join(training_dir_remapped, f"{category}_{split}.npy")
    shutil.copy(images_src, images_dst)
    logging.info(f"Copied {images_src} to {images_dst}")

    logging.info(f"finished remap_labels_and_copyimages for {category} {split} \n")

def combine_shuffle_image_label_data(training_dir_remapped, split):

    logging.info(f"started copy_shuffle_image_label_data for {split} \n")

    # Load remapped data
    herbveg_images = np.load(f"{training_dir_remapped}/HerbaceousVegetation_{split}.npy")
    herbveg_labels = np.load(f"{training_dir_remapped}/HerbaceousVegetation_{split}_labels.npy")
    residential_images = np.load(f"{training_dir_remapped}/Residential_{split}.npy")
    residential_labels = np.load(f"{training_dir_remapped}/Residential_{split}_labels.npy")
    
    # Combine data
    combined_images = np.concatenate((herbveg_images, residential_images), axis=0)
    combined_labels = np.concatenate((herbveg_labels, residential_labels), axis=0)
    
    # Shuffle data
    indices = np.arange(len(combined_labels))
    np.random.shuffle(indices)
    combined_images = combined_images[indices]
    combined_labels = combined_labels[indices]

    # Save combined and shuffled data
    np.save(f"{training_dir_remapped}/binary_{split}_images.npy", combined_images)
    np.save(f"{training_dir_remapped}/binary_{split}_labels.npy", combined_labels)

    logging.info(f"finished copy_shuffle_image_label_data for {split} \n")


def test_images_and_labels(data_dir, split):
    """
    Test the integrity of images and labels after preprocessing.

    Args:
    - data_dir (str): Directory containing the preprocessed data.
    - split (str): Dataset split ('train', 'val', or 'test').

    Returns:
    - None
    """
    try:
        logging.info(f"started test_images_and_labels for {split} \n")
        
        images = np.load(f"{data_dir}/binary_{split}_images.npy")
        labels = np.load(f"{data_dir}/binary_{split}_labels.npy")

        # Check shapes
        logging.info(f"{split.capitalize()} images shape: {images.shape}")
        logging.info(f"{split.capitalize()} labels shape: {labels.shape}")

        # Check if the number of images matches the number of labels
        assert images.shape[0] == labels.shape[0], "Mismatch in number of images and labels!"

        # Check range of pixel values
        logging.info(f"Pixel value range for {split} images: {images.min()} to {images.max()}")

        # Check unique label values
        unique_labels = np.unique(labels)
        logging.info(f"Unique labels in {split} set: {unique_labels}")

        # Display a few sample labels and check image-label alignment
        for i in range(5):
            logging.info(f"Sample {i} label: {labels[i]}")

        logging.info(f"finished test_images_and_labels for {split} \n")

    except Exception as e:
        logging.error(f"Error testing images and labels for {split} split: {e}")
        raise

    
def main():
    try:
        # Logging configuration
        log_dir = "../logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"03_E_remap_labels_combine_data_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        
        logging.info("\n")
        logging.info("called via 03_E_remap_labels_combine_shuffle_data.py...\n")
        logging.info(" Script Started ...\n")
        logging.info("This script will remap the labels, combine and shuffle the data for the two chosen categories: Residential and HerbaceousVegetation.\n")

        # Check if image size is passed as an argument
        if len(sys.argv) != 2:
            print("Usage: python 03_E_remap_labels_combine_shuffle_data.py <image_size>")
            sys.exit(1)

        # Get image size from the command-line argument
        image_size = int(sys.argv[1])
        logging.info(f"image_size is {image_size}\n")

        # Define paths
        training_dir = f"../data/training_data_{image_size}"
        training_dir_remapped = f"../data/training_data_remapped_binary_{image_size}"
        os.makedirs(log_dir, exist_ok=True)
        
        # Remap labels and save
        categories = ["Residential", "HerbaceousVegetation"]
        splits = ["train", "val", "test"]

        # remap_labels_and_copy_images
        for category in categories:
            for split in splits:
                remap_labels_and_copy_images(training_dir, training_dir_remapped, category, split)

        # combine_shuffle_image_label_data
        for category in categories:
            for split in splits:
                combine_shuffle_image_label_data(training_dir_remapped, split)
        
        #time to do some checks
        for split in splits:
            test_images_and_labels(training_dir_remapped, split)


    except Exception as e:
        logging.error(f"Error during dataset preparation: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Script failed: {e}")
