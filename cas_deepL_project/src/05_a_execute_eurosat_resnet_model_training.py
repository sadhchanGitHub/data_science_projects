import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import time
import os
from datetime import datetime
import sys

import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt

timestamp = int(time.time())


#import modules
from define_eurosat_resnet_model import eurosat_resnet_model_with_augmentation, eurosat_resnet_model_with_augmentation_l2
from define_eurosat_resnet_model import train_eurosat_resnet_model, visualize_eurosat_resnet_model_training

def chk_gpu_config():
    # Configure GPU settings (restrict TensorFlow to use only the first GPU)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)  # Optional: Enable memory growth
            logging.info("GPU configured successfully.")
        except RuntimeError as e:
            logging.info(f"Error configuring GPU: {e}")



def calculate_total_samples(training_dir, categories, batch_size, split_ratio=0.8):
    """
    Calculate total samples, training, and validation split sizes for each category.

    Args:
    - training_dir (str): Path to the training data directory.
    - categories (list): List of category names.
    - split_ratio (float): Ratio for training set (default: 0.8).

    Returns:
    - total_samples (int): Total number of samples available.
    - train_samples_per_category (int): Number of training samples per category.
    - val_samples_per_category (int): Number of validation samples per category.
    - steps_per_epoch (int): Number of steps per epoch for training.
    - validation_steps (int): Number of steps for validation.
    """
    total_samples = sum([len(np.load(os.path.join(training_dir, f"{category}_train.npy"))) for category in categories])
    logging.info(f"Total samples available: {total_samples}")

    # Split into training and validation samples
    num_train_samples = int(total_samples * split_ratio)
    num_val_samples = total_samples - num_train_samples

    logging.info(f"num_train_samples: {num_train_samples}, num_val_samples: {num_val_samples}\n")

    # Calculate samples per category
    train_samples_per_category = num_train_samples // len(categories)
    val_samples_per_category = num_val_samples // len(categories)

    logging.info(f"train_samples_per_category: {train_samples_per_category}, val_samples_per_category: {val_samples_per_category}\n")

    # Calculate steps per epoch
    steps_per_epoch = max(1, (train_samples_per_category * len(categories)) // batch_size)
    validation_steps = max(1, (val_samples_per_category * len(categories)) // batch_size)

    logging.info(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}\n")

    return total_samples, train_samples_per_category, val_samples_per_category, steps_per_epoch, validation_steps

def classification_data_generator(training_dir, categories, split, batch_size, samples_per_category):
    """
    Generator for loading images and labels in batches.
    """
    image_files = []
    label_files = []

    # Collect image and label file paths for all categories
    for category in categories:
        image_files += [os.path.join(training_dir, f"{category}_{split}.npy")]
        #label_files += [os.path.join(training_dir, f"{category}_{split}_labels_one_hot.npy")]
        label_files += [os.path.join(training_dir, f"{category}_{split}_labels.npy")]

    while True:
        for i in range(0, len(image_files)):
            images = np.load(image_files[i])[:samples_per_category]
            labels = np.load(label_files[i])[:samples_per_category]

            # Yield data in batches
            for j in range(0, len(images), batch_size):
                batch_images = images[j:j + batch_size]
                batch_labels = labels[j:j + batch_size]
                yield np.array(batch_images), np.array(batch_labels)



def main():
    try:

        # Logging configuration
        log_dir = "../logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"05_a_execute_eurosat_resnet_model_training_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        
        logging.info("\n")
        
        logging.info("called via 05_a_execute_eurosat_resnet_model_training.py...\n")
        logging.info(" Script Started ...\n")
        logging.info("This script will train the resnet model defined in define_eurosat_resnet_model.py \n")

        # Check if image size is passed as an argument
        if len(sys.argv) != 2:
            print("Usage: python 05_a_execute_eurosat_resnet_model_training.py <image_size>")
            sys.exit(1)
    
        # Get image size from the command-line argument
        image_size = int(sys.argv[1])
        logging.info(f"image_size is {image_size} \n ")

        # call gpu usage
        chk_gpu_config()
        
        batch_size = 32
        logging.info(f"batch_size is {batch_size} \n ")

        # Usage example
        training_dir = f"../data/training_data_{image_size}"
        # categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]
        categories = ["HerbaceousVegetation", "Residential", "AnnualCrop"]
        
        # Call the function to calculate sample sizes and steps
        total_samples, train_samples_per_category, val_samples_per_category, steps_per_epoch, validation_steps = calculate_total_samples(
            training_dir, categories, batch_size=batch_size
        )
        
        # Define data generators
        train_generator = classification_data_generator(
            training_dir, categories, split="train", batch_size=batch_size, samples_per_category=train_samples_per_category
        )
        val_generator = classification_data_generator(
            training_dir, categories, split="val", batch_size=batch_size, samples_per_category=val_samples_per_category
        )

        # Check a batch from the train generator
        batch_images, batch_labels = next(train_generator)
        
        logging.info(f"Batch images shape: {batch_images.shape}")
        logging.info(f"Batch labels shape: {batch_labels.shape}")


        logging.info(f"Data generators created with {steps_per_epoch} steps per epoch "
                     f"and {validation_steps} validation steps.")
    except Exception as e:
        logging.error(f"Error during dataset preparation: {e}")
        raise

    resnet_model_name = "eurosat_resnet_model_with_augmentation_L2"
    try:
        # Model parameters
        input_shape = (224, 224, 3)  # Shape of input images
        num_classes = 6  # Number of categories
        # Define model name beforehand to avoid referencing issues
        
        # Create and compile the model
        #model = create_cnn_eurosat_model(input_shape, num_classes)
        
        #not with data augmentation
        # resnet_model  = eurosat_resnet_model_with_augmentation(input_shape, num_classes)
        resnet_model  = eurosat_resnet_model_with_augmentation_l2(input_shape, num_classes)
        
        logging.info(f"model_name is {resnet_model_name} \n")
        # Print model summary
        resnet_model.summary()
    except Exception as e:
        logging.error(f"Error {resnet_model_name} model: {e}")
        raise

    try:
        # Train the model
        history, history_fine = train_eurosat_resnet_model(log_file, resnet_model_name, resnet_model, train_generator, steps_per_epoch, val_generator, validation_steps)

        logging.info(f"Model {resnet_model_name} training completed.")
    except Exception as e:
        logging.error(f"Error Model {resnet_model_name} during training: {e}")
        raise


    try:
    # Save full model
        resnet_model.save(f'../models/{resnet_model_name}_{timestamp}.keras')
        logging.info(f"Full model saved to: ../models/{resnet_model_name}_{timestamp}.keras")
        
        
        # Save weights
        resnet_model.save_weights(f'../models/{resnet_model_name}_weights_{timestamp}.weights.h5')
        logging.info(f"Model weights saved to: ../models/{resnet_model_name}_weights_{timestamp}.weights.h5")
        
    
    except Exception as e:
        logging.error(f"Error during saving {resnet_model_name} model / weights: {e}")
        raise

    try:
        # Visualize training progress no finetuned model
        visualize_eurosat_resnet_model_training(resnet_model_name, history_var=history, fine_tuning=False, timestamp=timestamp)
        logging.info("05_a_execute_eurosat_resnet_model_training progress history (fine_tuning=False) visualization completed.")

        visualize_eurosat_resnet_model_training(resnet_model_name, history_var=history_fine, fine_tuning=True, timestamp=timestamp)
        logging.info("05_a_execute_eurosat_resnet_model_training progress history (fine_tuning=True) visualization completed.")

    except Exception as e:
        logging.error(f"Error during 05_a_execute_eurosat_resnet_model_training progress visualization completed: {e}")
        raise
        
    logging.info(f"Script Ended \n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Script failed: {e}")
    
