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
from define_eurosat_binary_classify_CNN_model import create_cnn_eurosat_binary_classify
from define_eurosat_binary_classify_CNN_model import train_cnn_eurosat_binary_classify, main_visualizations


def chk_gpu_config():
    # Configure GPU settings (restrict TensorFlow to use only the first GPU)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)  # Optional: Enable memory growth
            print("GPU configured successfully.")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")



def classification_data_generator(training_dir_remapped, split, batch_size):
    """
    Generates batches of data for binary classification.
    Args:
    - data_dir: Directory containing the combined data files.
    - split: Either 'train' or 'val'.
    - batch_size: Number of samples per batch.
    """
    images = np.load(os.path.join(training_dir_remapped, f"binary_{split}_images.npy"))
    labels = np.load(os.path.join(training_dir_remapped, f"binary_{split}_labels.npy"))
    
    # Ensure shuffling happens before yielding
    dataset_size = len(labels)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    while True:
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images = images[batch_indices]
            batch_labels = labels[batch_indices]
            yield batch_images, batch_labels

def calculate_total_samples(training_dir_remapped):
    """
    Calculates the total number of samples for training and validation.
    Args:
    - training_dir_remapped: Directory containing the combined data files.
    """
    train_images = np.load(os.path.join(training_dir_remapped, "binary_train_images.npy"))
    val_images = np.load(os.path.join(training_dir_remapped, "binary_val_images.npy"))
    return len(train_images), len(val_images)


def sanity_check(training_dir_remapped):
    train_images = np.load(os.path.join(training_dir_remapped, "binary_train_images.npy"))
    train_labels = np.load(os.path.join(training_dir_remapped, "binary_train_labels.npy"))
    logging.info(f"train_images shape: {train_images.shape}")
    logging.info(f"train_labels shape: {train_labels.shape}")
    logging.info(f"Sample labels: {train_labels[:5]}")

    val_images = np.load(os.path.join(training_dir_remapped, "binary_val_images.npy"))
    val_labels = np.load(os.path.join(training_dir_remapped, "binary_val_labels.npy"))
    logging.info(f"val_images shape: {val_images.shape}")
    logging.info(f"val_labels shape: {val_labels.shape}")
    logging.info(f"Sample labels: {val_labels[:5]}")    

def main():
    try:
        
        # Logging configuration
        log_dir = "../logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"06_a_execute_cnn_eurosat_binary_classify_training_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        
        logging.info("\n")
        
        logging.info("called via 06_a_execute_cnn_eurosat_binary_classify_training.py...\n")
        logging.info(" Script Started ...\n")
        logging.info("This script will train the cnn model defined in define_eurosat_binary_classify_CNN_model.py \n")

        # Check if image size is passed as an argument
        if len(sys.argv) != 2:
            print("Usage: python 06_a_execute_cnn_eurosat_binary_classify_training.py <image_size>")
            sys.exit(1)
    
        # Get image size from the command-line argument
        image_size = int(sys.argv[1])
        logging.info(f"image_size is {image_size} \n ")

        # call gpu usage
        chk_gpu_config()

        training_dir_remapped = f"../data/training_data_remapped_binary_{image_size}"
        logging.info(f"training_dir_remapped is {training_dir_remapped} \n ")
        batch_size = 32
        logging.info(f"batch_size is {batch_size} \n ")

        # Sanity check
        sanity_check(training_dir_remapped)

        # Calculate total samples
        train_samples, val_samples = calculate_total_samples(training_dir_remapped)
    
        # Data generators
        train_generator = classification_data_generator(training_dir_remapped, "train", batch_size)
        val_generator = classification_data_generator(training_dir_remapped, "val", batch_size)
    
        steps_per_epoch = train_samples // batch_size
        validation_steps = val_samples // batch_size

        logging.info(f"steps_per_epoch is {steps_per_epoch}")
        logging.info(f"validation_steps is {validation_steps} \n ")

    except Exception as e:
        logging.error(f"Error during dataset preparation: {e}")
        raise

    model_name = "create_cnn_eurosat_binary_classify"
    logging.info(f"model_name is {model_name} \n")    
    #model related code starts here
    try:
        # Model parameters
        input_shape = (image_size, image_size, 3)  # Shape of input images
        num_classes = 2  # Number of categories
        
        # Create and compile the model
        model = create_cnn_eurosat_binary_classify(input_shape, num_classes)
        logging.info(f"Type of model: {type(model)}")  # Should print <class 'tensorflow.keras.Model'> or similar
       
        
        # Print model summary
        model.summary()
    except Exception as e:
        logging.error(f"Error {model_name} model: {e}")
        raise

    try:
        # Train the model
        history = train_cnn_eurosat_binary_classify(model_name, model, train_generator, steps_per_epoch, val_generator, validation_steps)
        logging.info(f"Model {model_name} training completed.")
    except Exception as e:
        logging.error(f"Error Model {model_name} during training: {e}")
        raise


    try:
    # Save full model
        model.save(f'../models/{model_name}_{timestamp}.keras')
        logging.info(f"Full model saved to: ../models/{model_name}_{timestamp}.keras")
        
        # Save weights
        model.save_weights(f'../models/{model_name}_weights_{timestamp}.weights.h5')
        logging.info(f"Model weights saved to: ../models/{model_name}_weights_{timestamp}.weights.h5")
        
    except Exception as e:
        logging.error(f"Error during saving {model_name} model / weights: {e}")
        raise

    # call the visualizations part
    try:
        # Load validation data
        val_images = np.load(os.path.join(training_dir_remapped, "binary_val_images.npy"))
        val_labels = np.load(os.path.join(training_dir_remapped, "binary_val_labels.npy"))
    
        # Call main visualizations
        main_visualizations(history, timestamp, model_name, model, val_images, val_labels)
    
    except Exception as e:
        logging.error(f"Error during {model_name} visualizations: {e}")
        raise
    



    logging.info(f"Script Ended at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Script failed: {e}")
    
