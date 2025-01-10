import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import time
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt

timestamp = int(time.time())


#import modules
#from define_eurosat_CNN_model import create_cnn_eurosat_model
from define_eurosat_CNN_model import train_cnn_eurosat_model, visualize_eurosat_cnn_model_training
from define_eurosat_CNN_model import create_cnn_eurosat_model_with_augmentation


# Logging configuration
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"04_b_execute_CNN_model_{int(datetime.now().timestamp())}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

print("\n")

print(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


# Configure GPU settings (restrict TensorFlow to use only the first GPU)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)  # Optional: Enable memory growth
        print("GPU configured successfully.")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")



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
    print(f"Total samples available: {total_samples}")

    # Split into training and validation samples
    num_train_samples = int(total_samples * split_ratio)
    num_val_samples = total_samples - num_train_samples

    print(f"num_train_samples: {num_train_samples}, num_val_samples: {num_val_samples}\n")

    # Calculate samples per category
    train_samples_per_category = num_train_samples // len(categories)
    val_samples_per_category = num_val_samples // len(categories)

    print(f"train_samples_per_category: {train_samples_per_category}, val_samples_per_category: {val_samples_per_category}\n")

    # Calculate steps per epoch
    steps_per_epoch = max(1, (train_samples_per_category * len(categories)) // batch_size)
    validation_steps = max(1, (val_samples_per_category * len(categories)) // batch_size)

    print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}\n")

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
        label_files += [os.path.join(training_dir, f"{category}_{split}_labels_one_hot.npy")]

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
        
        batch_size = 32

        # Usage example
        training_dir = "../data/training_data"
        #categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]
        #switching to binary classification as for all categries not working-
        categories = ["Highway", "HerbaceousVegetation"]
        
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
        
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")


        logging.info(f"Data generators created with {steps_per_epoch} steps per epoch "
                     f"and {validation_steps} validation steps.")
    except Exception as e:
        logging.error(f"Error during dataset preparation: {e}")
        raise

    try:
        # Model parameters
        input_shape = (256, 256, 3)  # Shape of input images
        num_classes = 2  # Number of categories
        
        # Create and compile the model
        # model = create_cnn_eurosat_model(input_shape, num_classes)

        model = create_cnn_eurosat_model_with_augmentation(input_shape, num_classes)
        print(f"Type of model: {type(model)}")  # Should print <class 'tensorflow.keras.Model'> or similar
       

        model_name = "create_cnn_eurosat_model_with_augmentation"
        print(f"model_name is {model_name} \n")
        # Print model summary
        model.summary()
    except Exception as e:
        logging.error(f"Error create_cnn_eurosat_model model: {e}")
        raise

    try:
        # Train the model
        history = train_cnn_eurosat_model(model_name, model, train_generator, steps_per_epoch, val_generator, validation_steps)
        logging.info(f"Model {model_name} training completed.")
    except Exception as e:
        logging.error(f"Error Model {model_name} during training: {e}")
        raise


    try:
    # Save full model
        model.save(f'../models/{model_name}_{timestamp}.keras')
        logging.info(f"Full model saved to: ../models/{model_name}_{timestamp}.keras")
        print(f"Full model saved to: ../models/{model_name}_{timestamp}.keras")
        
        # Save weights
        model.save_weights(f'../models/{model_name}_weights_{timestamp}.weights.h5')
        logging.info(f"Model weights saved to: ../models/{model_name}_weights_{timestamp}.weights.h5")
        print(f"Model weights saved to: ../models/{model_name}_weights_{timestamp}.weights.h5")
    
    except Exception as e:
        logging.error(f"Error during saving {model_name} model / weights: {e}")
        raise

    try:
        # Visualize training progress
        visualize_eurosat_cnn_model_training(history, timestamp, model_name)
        logging.info("Training progress visualization completed.")
    
    except Exception as e:
        logging.error(f"Error during Training progress visualization completed: {e}")
        raise
        
    print(f"Ended at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Script failed: {e}")
    
