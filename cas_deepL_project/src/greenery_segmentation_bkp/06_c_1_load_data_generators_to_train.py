import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from unet_model_definition import unet_mode_detect_greenery
from logging_config import setup_logging
import cv2
import os
import random

# Initialize logging
setup_logging(log_file_path="../logs/unet_training.log")

# Dataset paths
training_data_dir = "../data/training_data"

# Categories for segmentation
#  categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]
categories = ["AnnualCrop"]

def data_generator(image_dir, mask_dir, batch_size):
    """
    Generator that yields image and mask batches for training.
    Skips batches where all masks are empty.
    """
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    while True:
        for i in range(0, len(image_files), batch_size):
            batch_images = []
            batch_masks = []
            for j in range(i, min(i + batch_size, len(image_files))):
                # Load image and corresponding mask
                image = np.load(os.path.join(image_dir, image_files[j]))
                mask = np.load(os.path.join(mask_dir, mask_files[j]))
                
                # Skip if the entire mask batch is empty
                if np.max(mask) == 0:
                    logging.info(f"Skipping batch {mask_files[j]}: Empty mask.")
                    continue

                batch_images.append(image)
                batch_masks.append(mask)
            
            if len(batch_images) > 0:  # Ensure non-empty batches
                yield np.array(batch_images), np.array(batch_masks)


def main():
    try:
        # Parameters
        batch_size = 1
        target_shape = (512, 512)  # Adjusted to match model output shape
        num_classes = 2
        sample_size = 2

        # Paths for images and masks
        train_image_paths = [os.path.join(training_data_dir, "train", f"{cat}_train.npy") for cat in categories]
        train_mask_paths = [os.path.join(training_data_dir, "train", f"{cat}_train_masks_combined.npy") for cat in categories]
        val_image_paths = [os.path.join(training_data_dir, "val", f"{cat}_val.npy") for cat in categories]
        val_mask_paths = [os.path.join(training_data_dir, "val", f"{cat}_val_masks_combined.npy") for cat in categories]

        # Load and sample training data
        logging.info("Sampling training data...")
        x_train, y_train = load_and_sample_data(train_image_paths, train_mask_paths, target_shape, num_classes, sample_size)
        
        # Load and sample validation data
        logging.info("Sampling validation data...")
        x_val, y_val = load_and_sample_data(val_image_paths, val_mask_paths, target_shape, num_classes, sample_size)

        logging.info(f"Training data: Images={x_train.shape}, Masks={y_train.shape}")
        logging.info(f"Validation data: Images={x_val.shape}, Masks={y_val.shape}")
    except Exception as e:
        logging.error(f"Error during dataset preparation: {e}")
        raise

    # Initialize U-Net model
    try:
        logging.info("Initializing the U-Net model...")
        model = unet_mode_detect_greenery(input_size=(*target_shape, 3))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary(print_fn=logging.info)
        logging.info("U-Net model initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing the U-Net model: {e}")
        raise

    # Setup callbacks
    try:
        logging.info("Setting up callbacks...")
        checkpoint_path = "unet_best_model.keras"
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
        csv_logger = CSVLogger('training_log.csv', append=True)
        logging.info("Callbacks setup complete.")
    except Exception as e:
        logging.error(f"Error setting up callbacks: {e}")
        raise

    # Train the model
    try:
        logging.info("Starting model training...")
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=2,
            batch_size=batch_size,
            callbacks=[checkpoint, csv_logger],
            verbose=1
        )
        logging.info(f"Training complete. Best model saved to {checkpoint_path}.")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise


# Entry point
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Script failed: {e}")
